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
from torch_geometric.data import Data
import json
import logging
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F



class ZeroShot:
    def __init__(self, query_path :str, gnn_model_path: str, GNN_class, vector_db : VectorDatabase):
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
        self.GNN_model.load_state_dict(torch.load(gnn_model_path, map_location= self.device))
        self.GNN_model.to(self.device)
        logging.basicConfig(    
            filename=config.ZERO_SHOT_ERROR_LOG,  # Log file name
            level=logging.ERROR,       # Log only errors and above
            format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        )


    def get_qep_graph(self, command : str):
        qep, _, _, _, error = self.db.getQep(command)
        #print(qep)
        g = Graph()
        g.parseQep(qep)
        return g, qep, error

    def run_improvedqueryschedule(self, query: str, num_alts: int = 3):
        embeddings = None
        g, qep, error = self.get_qep_graph(query)
        if error:
            print(f'the following error occured - {error} while parsing the query \n {query}')
            raise error
        hints, alt_queries, error = self.get_alt_query(query, g.nodes, g.edges, 3)
        if error:
            print(f'the following error occured - {error} while getting alternate queries for the following query:\n {query}')
            raise error
        improved_count = 0
        print('improving the query...')
        queries = [query]
        queries.extend(alt_queries)
        query_time, improved_list, alt_query_time_list, error = self.db.compare_n_queries(queries, num_alts + 1)
        if error:
            print(f'the following error occured - {error} while parsing the query \n {query}')
        for i in range(num_alts):
            improved = improved_list[i]
            alt_query_time = alt_query_time_list[i]
            if improved:
                improved_count += 1
                time_improved = (query_time - alt_query_time) / query_time
                config_settings = json.dumps(self.node_embedder.extractConfigurations(config.DATABASE_CONFIG))
                metadata = pd.DataFrame({'query': [query], 'hints': [hints[i]], 'time_improved': [time_improved], 'database_config' : [config_settings]})
                if embeddings is None:
                    embeddings = self.embed(query)
                self.vector_db.add_vectors(embeddings = embeddings.detach().cpu().numpy(), metadata=metadata)
        # Append to a simple text file
        with open(config.ZERO_SHOT_STATUS, "a") as f:
            f.write(f"Added {improved_count} alternative queries for the original query: {query}\n")


    def get_alt_query(self, command, nodes, edges, num_alts):
        try:
            hints, mod_queries = self.llm_model.get_n_hints(command, nodes, edges, num_alts)
            return hints, mod_queries, None
        except Exception as e:
            print(f'An unexpected error occured {e}')
            return None, None, e
    
    def embed(self, query):
        graph, _, _ = self.get_qep_graph(query)
        query_embedding = torch.tensor(self.node_embedder.batch_processor(data=[graph.nodes])[0], dtype=torch.float)
        query_tree = Data(query_embedding, torch.tensor(graph.edges, dtype=torch.long)).to(self.device)
        query_tree.batch = torch.zeros(query_tree.x.shape[0], dtype=torch.long).to(self.device)
        query_embedding = self.GNN_model(query_tree.x, query_tree.edge_index, query_tree.batch)
        return query_embedding
    
    def run(self):
        self.db.connect()
        for query in self.queries:
            try:
                self.run_improvedqueryschedule(query)
            except Exception as e:
                logging.error(f"Error executing query: {query}", exc_info=True)  # Logs the error with traceback
        self.db.close()

if __name__ == '__main__':
    class GNNEncoder(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNEncoder, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            # Global mean pooling to get a graph-level representation
            x = global_mean_pool(x, batch)
            return x
    
    GNN = GNNEncoder(48, 64, 32)

    vector_db = VectorDatabase(config.VD_INDEX, 32, 'cosine', config.METADATA, ['query', 'hints', 'time_improved', 'database_config'], None) 
    zero_shot_pipeline = ZeroShot(config.TPCH, config.GNN_MODEL, GNN, vector_db)
    zero_shot_pipeline.run()
