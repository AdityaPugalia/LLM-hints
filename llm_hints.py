from db import Database
import config
from qep import Graph
from LLM import LLM
import pandas as pd
from vector_database import VectorDatabase
import torch
from embeddings import nodeEmbedder
import ipynb
from ipynb.fs.full.GNN_embeddings import GNNEncoder
from torch_geometric.data import Data
import pickle



class llm_hints:
    def __init__(self, gnn_model_path: str, GNN_class : GNNEncoder, vector_db : VectorDatabase, learn : bool = False):
        self.db = Database(user= config.USER, dbname= config.DBASE)
        self.vector_db = vector_db
        self.llm_model = LLM()
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_embedder = nodeEmbedder(1, self.device, False)
        self.GNN_model = GNN_class
        self.GNN_model.load_state_dict(torch.load(gnn_model_path))
        self.GNN_model.to(self.device)
        self.learn = learn

    def get_qep_graph(self, command : str):
        qep, _, _, _, error = self.db.getQep(command)
        #print(qep)
        g = Graph()
        g.parseQep(qep)
        return g, qep, error
    
    def embed(self, graph):
        query_embedding = torch.tensor(self.node_embedder.batch_processor(nodes=graph.nodes)[0], dtype=torch.float)
        query_tree = Data(query_embedding, torch.tensor(graph.edges, dtype=torch.long)).to(self.device)
        query_embedding = self.GNN_model(query_tree)
        return query_embedding
    
    def improve_query(self, query):
        try:
            graph, _, error = self.get_qep_graph(query)
            if error:
                raise error
            embedding = self.embed(graph)
            data = self.vector_db.search(query_embedding= embedding, k = 5)
            self.llm_model.create_demonstrations(data)
            hint, mod_query = self.llm_model.getHints(query, graph.nodes, graph.edges, True)
            return hint, mod_query, embedding
        except Exception as e:
            raise e
    
    def execute_improved_query(self, query):
        try:
            hint, mod_query, query_embedding = self.improve_query(query)
            improved, query_time, mod_query_time, error = self.db.compareQuery(query, mod_query, 1)
            if error:
                raise error
            return hint, query_embedding, improved, query_time, mod_query_time
        except Exception as e:
            raise e

    def run_test_schedule(self, query_path, stat_path):
        queries = pd.read_csv(query_path)['queries']
        stats = pd.DataFrame(columns= ['query', 'hint', 'query time', 'mod-query time', 'improved', 'time diff'])
        self.db.connect()
        for query in queries:
            try:    
                hint, query_embedding, improved, query_time, mod_query_time = self.execute_improved_query(query)
                stats.loc[len(stats)] = [query, hint, query_time, mod_query_time, improved, query_time - mod_query_time]
                if self.learn and improved:
                    self.vector_db.add_vectors(embeddings= query_embedding, metadata= stats[['query', 'hint', 'time diff']].iloc[[-1]])
            except Exception as e:
                print (f'the following error {e} occured while executing the query: \n {query}')
        self.db.close()
        with open(stat_path, 'wb') as file:
            pickle.dump(stats, file)
        



            


    