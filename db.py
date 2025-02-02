import psycopg2
import json
import time
import os
from typing import Tuple
import config


class Database:
    def __init__(self, host = 'localhost', port = 5432, dbname = None, user = 'postgres', password = None):
        self.db_name = dbname
        self.db_user = user
        self.db_password = password
        self.db_host = host
        self.db_port = port
        self.conn = None
        self.cursor = None

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

    def close(self):
        """
        Close the database connection
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def executeQuery(self, query, params=None):
        """
        Execute a query and return the result
        """
        try:
            start_time = time.time()
            self.cursor.execute(query, params)

            if self.cursor.description:
                result = self.cursor.fetchall()
                row_count = self.cursor.rowcount
            else:
                self.conn.commit()
                result = None
                row_count = 0

            execution_time = time.time() - start_time
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
    
    def runExecutions(self, query) -> Tuple[float, Exception]:
        avg_time = 0
        for i in range(5):
            result, execution_time, _, error = self.executeQuery(query)
            if error:
                return None, error
            avg_time += execution_time
        avg_time /= 5
        return avg_time, None
        
    def compareQuery(self, query, alt_query):
        query_avg, error = self.runExecutions(query)
        if error:
            return None, None, None, error
        alt_query_avg, error = self.runExecutions(alt_query)
        if error:
            return None, None, None, error
        return alt_query_avg < query_avg, query_avg, alt_query_avg, None

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

    def get_config_settings(self, output_folder='data', output_file='config_settings.json'):
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

if __name__ == "__main__":
    db = Database(user= config.USER, dbname= config.DBASE)
    db.connect()
    db.extract_schema()
    db.close()

