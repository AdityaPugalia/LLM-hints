from db import Database
import config
from qep import Graph
import psycopg2
from LLM import LLM
import torch.nn.functional as F
from vector_database import VectorDatabase
from zero_shot_pipeline import ZeroShot
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool
import torch

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
    command = 'SELECT COUNT(*) FROM customer INNER JOIN orders ON customer.c_custkey = orders.o_custkey WHERE o_orderdate > \'2015-01-01\';'
    print(command, "\n\n")
    #avg_time, error = db.runExecutions(command)
    # if not error:
    #     print(avg_time)
    g, qep, error = getQepGraph(command)
    llm = LLM()
    hints, mod_query = llm.getHints(command, g.nodes, g.edges, )
    print(mod_query, "\n\n")
    g_mod, mod_query_qep, error = getQepGraph(mod_query)
    print(mod_query_qep, '\n\n')
    improved, query_time, mod_query_time, error = db.compareQuery(command, mod_query, num_runs = 1)
    print(f"The query was improved: {improved}. The original query run-time was : {query_time} and the modified query run time was : {mod_query_time}\n")
    #db.get_config_settings()
    # command = 'SELECT elem_count_histogram FROM pg_stats WHERE tablename = \'customer\''
    # result, _, _, _ = db.executeQuery(command)
    # print(result[3])    
    db.close()

# if __name__ == '__main__':
#     class GNNEncoder(torch.nn.Module):
#         def __init__(self, input_dim, hidden_dim, output_dim):
#             super(GNNEncoder, self).__init__()
#             self.conv1 = GCNConv(input_dim, hidden_dim)
#             self.conv2 = GCNConv(hidden_dim, output_dim)

#         def forward(self, x, edge_index, batch):
#             x = self.conv1(x, edge_index)
#             x = F.relu(x)
#             x = self.conv2(x, edge_index)
#             # Global mean pooling to get a graph-level representation
#             x = global_mean_pool(x, batch)
#             return x
    
#     GNN = GNNEncoder(48, 64, 32)
#     vd = VectorDatabase('/mnt/newpart/lizhaodonghui/hint/LLM hints/data/dummy_index.faiss', 32, 'cosine', '/mnt/newpart/lizhaodonghui/hint/LLM hints/data/Dummy_Queries.sqlite' , ['query'], None)
#     zero_shot_pipeline = ZeroShot('/mnt/newpart/lizhaodonghui/hint/LLM hints/traindataset/queries_tpch_train.csv', '/mnt/newpart/lizhaodonghui/hint/LLM hints/models/GNNQueryEncoder.pt', GNN, vd)
#     zero_shot_pipeline.db.connect()
#     queries = pd.read_csv('/mnt/newpart/lizhaodonghui/hint/LLM hints/traindataset/queries_tpch_train.csv')['queries'][25]
#     # count = 0
#     # for query in queries:
#     #     try:
#     #         embedding = zero_shot_pipeline.embed(query)
#     #         print(embedding, '\n')
#     #         vd.add_vectors(None, embedding.detach().cpu().numpy(), pd.DataFrame([query], columns= ['query']))
#     #     except Exception as e:
#     #         print(e)
#     #         continue
#     # zero_shot_pipeline.db.close()
#     embedding = zero_shot_pipeline.embed(queries)
#     df = pd.DataFrame([queries], columns= ['query'])
#     print(queries, '\n\n')
#     results = vd.search(query_embedding=embedding.detach().cpu().numpy(), threshold= 0.6)
#     print(results)