from db import Database
import config
from qep import Graph
import psycopg2
from LLM import LLM
import pandas as pd
from vector_database import VectorDatabase
import torch
from embeddings import nodeEmbedder
import ipynb
from ipynb.fs.full.GNN_embeddings import GNNEncoder
from torch_geometric.data import Data



class ZeroShot:
    def __init__(self, query_path :str, gnn_model_path: str, GNN_class : GNNEncoder, vector_db : VectorDatabase):
        self.db = Database(user= config.USER, dbname= config.DBASE)
        self.queries = pd.read_csv(query_path)['queries']
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


    def get_qep_graph(self, command : str):
        qep, _, _, _, error = self.db.getQep(command)
        #print(qep)
        g = Graph()
        g.parseQep(qep)
        return g, qep, error

    def run_improvedqueryschedule(self, query: str):
        embeddings = None
        g, qep, error = self.get_qep_graph(query)
        if error:
            print(f'the following error occired - {error} while parsing the query \n {query}')
            raise error
        hints, alt_queries, error = self.get_alt_query(query, g.nodes, g.edges)
        if error:
            print(f'the following error occured - {error} while getting alternate queries for the following query:\n {query}')
            raise error
        for i in range(len(hints)):
            improved, query_time, alt_query_time, error = self.db.compareQuery(query, alt_queries[i])
            if error:
                print(f'the following error occured - {error} while running the following alt query:\n {alt_queries[i]}')
                continue
            if improved:
                time_diff = query_time - alt_query_time
                metadata = pd.Dataframe({'query': query, 'hints': hints[i], 'time_diff': time_diff})
                if embeddings is not None:
                    embeddings = self.embed(query)
                self.vector_db.add_vectors(embeddings = embeddings, metadata=metadata)

    def get_alt_query(self, command, nodes, edges):
        try:
            hints, mod_queries = self.llm_model.get_n_hints(command, nodes, edges)
            return hints, mod_queries, None
        except Exception as e:
            print(f'An unexpected error occured {e}')
            return None, None, e
    
    def embed(self, query):
        graph, _, _ = self.get_qep_graph(query)
        query_embedding = torch.tensor(self.node_embedder.batch_processor(nodes=graph.nodes)[0], dtype=torch.float)
        query_tree = Data(query_embedding, torch.tensor(graph.edges, dtype=torch.long)).to(self.device)
        query_embedding = self.GNN_model(query_tree)
        return query_embedding
    
    def run(self):
        try:
            self.db.connect()
            for query in self.queries:
                self.run_improvedqueryschedule(query)
        except Exception as e:
            print(e)
        finally:
            self.db.close()
        pass

if __name__ == '__main__':
    pass
