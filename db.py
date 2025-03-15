import psycopg2
import json
import time
import os
from typing import Tuple
import config
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class Database:
    def __init__(self, host = 'localhost', port = config.PORT, dbname = None, user = 'postgres', password = None, query_timeout : int = 300):
        self.db_name = dbname
        self.db_user = user
        self.db_password = password
        self.db_host = host
        self.db_port = port
        self.conn = None
        self.cursor = None
        self.monitor_thread = threading.Thread(target=self.monitor_long_queries, daemon=True)
        self.monitor_thread_stop_event = threading.Event()
        self.query_timeout = query_timeout

    def connect(self):
        """
        Connect to the database
        """
        self.conn = psycopg2.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        self.cursor = self.conn.cursor()
        self.monitor_thread_stop_event.clear()
        self.monitor_thread.start()
        

    def close(self):
        """
        Close the database connection
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        if self.monitor_thread is not None:
            self.monitor_thread_stop_event.set()
            self.monitor_thread.join()

    def executeQuery(self, query, params=None, conn = None):
        """
        Execute a query and return the result
        """
        try:
            cursor = conn.cursor() if conn is not None else self.conn.cursor()
            start_time = time.time()
            cursor.execute(query, params)
            execution_time = time.time() - start_time

            if cursor.description:
                result = cursor.fetchall()
                row_count = cursor.rowcount
            else:
                self.conn.commit()
                result = None
                row_count = 0
            return result, execution_time, row_count, None
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            # Rollback to clear the transaction state
            print('rolling back transaction')
            self.conn.rollback()
            print('transaction rolled back')
            return None, 0, 0, str(e)


    def analyze(self):
        """
        Analyze the database
        """
        query = "ANALYZE;"
        return self.executeQuery(query)

    def getQep(self, query):
        """
        Get the query execution plan (QEP) for a query
        """
        query = f"EXPLAIN (FORMAT JSON) {query}"
        result, execution_time, _, error = self.executeQuery(query)
        if error:
            return None, None, None, None, error
        # Extract the total cost of the top-level plan
        qep_cost = None
        qep_rows = None
        if result is not None:
            qep_cost = result[0][0][0]["Plan"]["Total Cost"]
            qep_rows = result[0][0][0]["Plan"]["Plan Rows"]
        return result[0][0], qep_cost, qep_rows, execution_time, error
    
    def runExecutions(self, query: str, num_runs: int = 5) -> Tuple[float, Exception]:
        """
        Runs `num_runs` executions of the query in parallel and returns the average execution time.
        """
        avg_time = 0
        errors = []

        # Function to execute query with a new connection
        def run_query():
            conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            result, execution_time, _, error = self.executeQuery(query, conn = conn)
            conn.close()  # Ensure connection is closed
            if error:
                errors.append(error)
                return 0  # Return 0 execution time in case of error
            return execution_time

        # Execute queries in parallel
        with ThreadPoolExecutor(max_workers=num_runs) as executor:
            futures = [executor.submit(run_query) for _ in range(num_runs)]
            
            # Collect execution times
            execution_times = [future.result() for future in as_completed(futures)]

        # Compute average time
        if errors:
            return None, errors[0]  # Return the first error encountered

        avg_time = sum(execution_times) / num_runs
        return avg_time, None
        
    def compareQuery(self, query, alt_query, num_runs : int = 1):
        query_avg, error = self.runExecutions(query, num_runs)
        if error:
            return None, None, None, error
        alt_query_avg, error = self.runExecutions(alt_query, num_runs)
        if error:
            return None, None, None, error
        return alt_query_avg < query_avg, query_avg, alt_query_avg, None
    
    def compare_n_queries(self, queries, num_queries : int = 1, num_runs : int = 1):
        # Execute queries in parallel
        with ThreadPoolExecutor(max_workers=num_queries) as executor:
            futures = [executor.submit(self.runExecutions, queries[i], num_runs) for i in range(num_queries)]
            
            # Collect execution times
            results = [future.result() for future in futures]
        improved_list = []
        alt_query_time_list = []
        query_result = results.pop(0)
        if query_result[1] is not None:
            return None, None, None, query_result[1]
        else:
            query_avg = query_result[0]
        for result in results:
            if result[1] is not None:
                improved_list.append(False)
                alt_query_time_list.append(None)
                continue
            alt_query_time = result[0]
            alt_query_time_list.append(alt_query_time)
            improved = alt_query_time < query_avg
            improved_list.append(improved)
        
        return (query_avg, improved_list, alt_query_time_list, None)

    def getStatistics(self, table_names, output_folder = 'data', output_file='table_statistics.json'):
        """Retrieve statistics for the specified tables and save them to a file."""
        if not self.conn or not self.cursor:
            raise Exception("Database connection is not established. Call connect() first.")
        
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Combine folder and file name
        output_path = os.path.join(output_folder, output_file)


        statistics = {}
        
        try:
            for table in table_names:
                query = f"""
                SELECT 
                    schemaname,
                    tablename,
                    attname AS column_name,
                    null_frac,
                    avg_width,
                    n_distinct,
                    most_common_vals,
                    most_common_freqs,
                    histogram_bounds,
                    correlation
                FROM pg_stats
                WHERE tablename = %s;
                """
                self.cursor.execute(query, (table,))
                stats = self.cursor.fetchall()

                # Column names from the query result
                columns = [desc[0] for desc in self.cursor.description]
                table_stats = [dict(zip(columns, row)) for row in stats]

                statistics[table] = table_stats
            
            # Write statistics to a file in JSON format
            with open(output_path, 'w') as file:
                json.dump(statistics, file, indent=4)
            print(f"Statistics saved to {output_file}")
        except Exception as e:
            print(f"Error fetching statistics: {e}")
            raise


    def get_config_settings(self, output_folder=config.DATA_FOLDER_PATH, output_file='config_settings.json'):
        """Fetch specified PostgreSQL configuration settings and save to a file."""
        if not self.conn or not self.cursor:
            raise Exception("Database connection is not established. Call connect() first.")

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Combine folder and file name
        output_path = os.path.join(output_folder, output_file)

        # List of configuration parameters to extract
        config_params = ['shared_buffers', 'work_mem', 'hash_mem_multiplier', 'temp_file_limit']

        try:
            settings = {}
            for param in config_params:
                query = "SELECT name, setting, unit, context FROM pg_settings WHERE name = %s;"
                self.cursor.execute(query, (param,))
                result = self.cursor.fetchone()
                
                if result:
                    settings[param] = {
                        "name": result[0],
                        "setting": result[1],
                        "unit": result[2],
                        "context": result[3],
                    }
                else:
                    # If `hash_mem_multiplier` does not exist, set it to 1
                    if param == "hash_mem_multiplier":
                        settings[param] = {
                            "name": param,
                            "setting": "1",
                            "unit": None,
                            "context": "Manually Set",
                        }

            # Write settings to a file in JSON format
            with open(output_path, 'w') as file:
                json.dump(settings, file, indent=4)

            print(f"Configuration settings saved to {output_path}")

        except Exception as e:
            print(f"Error fetching configuration settings: {e}")
            raise

    def extract_schema(self, output_folder='data', output_file='database_schema.json'):
        """Extract table names and column names, and save them to a JSON file."""
        if not self.conn or not self.cursor:
            raise Exception("Database connection is not established. Call connect() first.")

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Combine folder and file name
        output_path = os.path.join(output_folder, output_file)

        schema_data = {}

        try:
            # Query to get table and column details
            query = """
            SELECT 
                table_name, 
                column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            # Group columns by table
            for table_name, column_name in results:
                if table_name not in schema_data:
                    schema_data[table_name] = []
                schema_data[table_name].append(column_name)

            # Write schema data to a JSON file
            with open(output_path, 'w') as file:
                json.dump(schema_data, file, indent=4)

            print(f"Database schema saved to {output_path}")
        except Exception as e:
            print(f"Error extracting schema: {e}")
            raise
    
    def get_indexes(self, output_folder=config.DATA_FOLDER_PATH, output_file='database_indexes.json'):
        """Retrieve all indexes for the database and save them to a JSON file."""
        if not self.conn or not self.cursor:
            raise Exception("Database connection is not established. Call connect() first.")

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Combine folder and file name
        output_path = os.path.join(output_folder, output_file)

        indexes = {}

        try:
            # Query to get indexes for all tables in the current database
            query = """
            SELECT
                t.relname AS table_name,
                i.relname AS index_name,
                array_to_string(array_agg(a.attname), ', ') AS column_names,
                ix.indisunique AS is_unique,
                ix.indisprimary AS is_primary
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = 'public'
            GROUP BY t.relname, i.relname, ix.indisunique, ix.indisprimary
            ORDER BY t.relname, i.relname;
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            # Organize data by table name
            for table_name, index_name, column_names, is_unique, is_primary in results:
                if table_name not in indexes:
                    indexes[table_name] = []
                indexes[table_name].append({
                    "index_name": index_name,
                    "columns": column_names,
                    "is_unique": is_unique,
                    "is_primary": is_primary
                })

            # Write indexes to a JSON file
            with open(output_path, 'w') as file:
                json.dump(indexes, file, indent=4)

            print(f"Database indexes saved to {output_path}")

        except Exception as e:
            print(f"Error extracting indexes: {e}")
            raise
    
    def monitor_long_queries(self):
        """
        Periodically checks for long-running queries and cancels them.
        """
        while not self.monitor_thread_stop_event.is_set():
            try:
                conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
                conn.autocommit = True
                cur = conn.cursor()

                # Query to find long-running queries
                cur.execute("""
                    SELECT pid, age(now(), query_start), query
                    FROM pg_stat_activity
                    WHERE state = 'active'
                    AND query_start < now() - interval '%s seconds';
                """, (self.query_timeout,))

                long_queries = cur.fetchall()

                for pid, duration, query in long_queries:
                    print(f"Cancelling query (PID: {pid}, Duration: {duration}): {query}")
                    cur.execute(f"SELECT pg_cancel_backend({pid})")

                cur.close()
                conn.close()

            except Exception as e:
                print(f"Error monitoring queries: {e}")

            # Check every 10 seconds
            time.sleep(10)
        print('exiting the monitor query thread')


if __name__ == "__main__":
    import pandas as pd
    db = Database(user= config.USER, dbname= config.DBASE, port= config.PORT, password= config.PASSWORD)
    db.connect()
    queries = pd.read_csv(config.TPCH)['queries']
    time.sleep(5)
    results = db.compare_n_queries([queries[2], queries[1], queries[1], queries[1]], 4)
    print(results)
    db.close()

