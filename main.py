from db import Database
import config
from qep import Graph
import psycopg2
from LLM import LLM

def getQepGraph(command : str):
    qep, _, _, _, error = db.getQep(command)
    #print(qep)
    g = Graph()
    g.parseQep(qep)
    return g, qep, error

def runImproveQuerySchedule(query: str):
    g, qep, error = getQepGraph(query)
    if error:
        return
    hint, alt_query, error = getAltQuery(query, g.nodes, g.edges)
    if error:
        return
    improved, query_time, alt_query_time, error = db.compareQuery(query, alt_query)
    if error:
        return
    if improved:
        pass
    
def getAltQuery(command, nodes, edges):
    try:
        llm = LLM()
        hint, mod_query = llm.getHints(command, nodes, edges)
        return hint, mod_query, None
    except Exception as e:
        print(f'An unexpected error occured {e}')
        return None, None, e

    # print('\n\n', hint, '\n\n', mod_query)

if __name__ == '__main__':
    db = Database(user= config.USER, dbname= config.DBASE)
    db.connect()
    table_names = ['customer', 'parts', 'region']
    command = 'SELECT COUNT(*co) FROM customer INNER JOIN orders ON customer.c_custkey = orders.o_custkey WHERE o_orderdate > \'2015-01-01\';'
    #avg_time, error = db.runExecutions(command)
    # if not error:
    #     print(avg_time)
    g, qep, error = getQepGraph(command)
    print(g.nodes, g.edges)
    print(qep)
    #db.get_config_settings()
    # command = 'SELECT elem_count_histogram FROM pg_stats WHERE tablename = \'customer\''
    # result, _, _, _ = db.executeQuery(command)
    # print(result[3])    
    db.close()