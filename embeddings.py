from transformers import BertTokenizer, BertModel
import torch
from qep import Graph
import json
from db import Database
import config
import numpy as np
import pandas as pd
import time
from PCA import PCAEmbeddingReducer
import pickle

class nodeEmbedder:
    def __init__(self, batch_size: int, device = torch.device('cpu'), pca_fit : bool = False):
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.config = self.extractConfigurations(config.DATABASE_CONFIG) 
        self.shared_buffers_inv = 1 / self.config['shared_buffers']
        self.work_mem_inv = 1 / self.config['work_mem']
        self.hash_mem_multiplier_inv = 1 / (self.config['work_mem'] * self.config['hash_mem_multiplier'])
        self.pca_fit = pca_fit

    def batch_processor(self, data):
        n = len(data)
        counter = 0
        out = []
        while counter < n:
            batch = data[counter: min(counter + self.batch_size, n)]
            out.extend(self.createBatchGraphFeatures(batch))
            counter += self.batch_size
            print(f'{counter} queries processed...')
        print('All queries procesed')
        return out
        
    def bertEmbedding(self, texts):
        
        # Tokenize input
        inputs = self.tokenizer(texts, return_tensors="pt", padding = True)

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.bert_model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().tolist()

        # Singular value embedding
        return cls_embedding
     
    def extractConfigurations(self, file_path):
        """
        Extract PostgreSQL configurations from a JSON file and convert to kilobytes.
        
        Parameters:
            file_path (str): Path to the JSON configuration file.
        
        Returns:
            dict: A dictionary with configuration names as keys and their values in KB.
        """
        with open(file_path, 'r') as file:
            config_data = json.load(file)
        
        config_in_kb = {}
        
        for key, config in config_data.items():
            setting = int(config["setting"])
            unit = config["unit"]
            
            # Convert to kilobytes
            if unit == "8kB":
                config_in_kb[key] = setting * 8  # 8kB to KB
            elif unit == "kB":
                config_in_kb[key] = setting      # Already in KB
            elif unit is None:  # Unit-less settings
                config_in_kb[key] = setting
            else:
                raise ValueError(f"Unsupported unit '{unit}' for {key}")
        
        return config_in_kb
    
    def reduce_embeddings(self, data, fit_transform : bool = True, variance = 0.9):
        reducer = PCAEmbeddingReducer(variance_threshold= variance)
        if self.pca_fit:
            data = reducer.fit(data, transform= fit_transform)
            reducer.save(config.PCA)
        else:
            reducer.load(config.PCA)
            data = reducer.transform(data)
        return data

    def createBatchGraphFeatures(self, batch_of_queries):
        """
        Processes a batch of query trees into a single matrix of feature vectors.

        :param batch_of_queries: List of query trees, where each query tree is a list of nodes (dicts).
        :return: A tuple (features, query_ids), where:
                - `features` is a NumPy matrix where each row is a feature vector for a node.
                - `query_ids` is a NumPy array mapping each row to a query tree index.
        """
        # Flatten all query trees into a single list of nodes and track query IDs
        flat_nodes = []
        query_ids = {}
        unique_queries = 0
        
        for i, query in enumerate(batch_of_queries):
            
            flat_nodes.extend(query)  # Flatten nodes
            query_ids[i] = len(query)  # Assign query index to each node
            unique_queries += 1

        # Convert to Pandas DataFrame for efficient vectorized operations
        df = pd.DataFrame(flat_nodes)
        # Extract numeric fields (Vectorized)
        cumulative_costs = df['Cummulative Cost'].to_numpy(dtype=float)
        planned_rows = df['Planned rows'].to_numpy(dtype=float)
        widths = df['Width'].to_numpy(dtype=float)

        # Compute planned row ratios (Vectorized)
        planned_row_ratios = planned_rows * widths * self.shared_buffers_inv

        # Compute label embeddings in batch mode
        labels = df['label'].to_list()
        label_embeddings = np.array(self.bertEmbedding(labels))  

        # Compute condition features (Batch Processing)
        join_condition_embedding = np.array(self.bertEmbedding(df['Join Condition'].fillna('').to_list()))  

        index_condition_embedding = np.array(self.bertEmbedding(df['Index Condition'].fillna('').to_list())) 

        # Compute filter embeddings (Batch Processing)
        filters = df['Filter'].fillna('').to_list()
        filter_embeddings = np.array(self.bertEmbedding(filters))  

        # Compute work memory ratios (Vectorized)

        # Create a mask for 'hash join' condition
        hash_mask = df['Parent Hash']
        
        # Compute work memory ratios using vectorized operations
        work_mem_ratios = np.where(
            hash_mask,
            planned_rows * widths * self.hash_mem_multiplier_inv,
            planned_rows * widths * self.work_mem_inv
        )

        # Combine all extracted features into a single matrix
        features = np.column_stack((
            label_embeddings,  # (n_nodes, embedding_dim)
            cumulative_costs.reshape(-1, 1),
            planned_row_ratios.reshape(-1, 1),
            join_condition_embedding, #(n_nodes, embedding_dim)
            index_condition_embedding, # (n_nodes, embedding_dim)
            filter_embeddings,  # (n_nodes, embedding_dim)
            work_mem_ratios.reshape(-1, 1)
        ))

        if self.pca_fit == False:
            features = self.reduce_embeddings(features)
            return self.formatted_output(features, query_ids, unique_queries)
        else:
            return [features]
    
    def formatted_output(self, features, query_ids, unique_queries):
        # Split the feature matrix into separate query trees
        query_tree_features = []
        counter = 0
        for q_id in range(unique_queries):
            add = query_ids[q_id]
            query_tree_features.append(features[counter: counter + add,])
            counter += add
        return query_tree_features
        
    def train_pca(self, data, variance = 0.9):
        output = self.batch_processor(data)
        output = np.vstack(output)
        # print(output)
        output = self.reduce_embeddings(output, False, variance)

    

if __name__ == '__main__':
    # commands = pd.read_csv('traindataset/queries_tpch_train.csv')['original_sql'][1:4]
    node_emb = nodeEmbedder(10, 'mps', False)
    # db = Database(user= config.USER, dbname= config.DBASE)
    # db.connect()
    # x = []
    # for query in commands:
    #     qtree, _, _, _, error = db.getQep(query)
    #     #filtering out invalid queries from the query set
    #     if error:
    #         continue
    #     G = Graph()
    #     G.parseQep(qtree)
    #     x.append(G.nodes)
    # db.close()

    # print(type(x[0]))
    # with open("traindataset/qtrees.pkl", "wb") as file:
    #     pickle.dump(x, file)
    with open("traindataset/qtrees.pkl", "rb") as file:
        x = pickle.load(file)
    print(len(x))
    start_time = time.time()
    # result = node_emb.train_pca(x[:500], 0.99)
    result = node_emb.batch_processor(x)
    #result = node_emb.reduce_embeddings(result)
    with open("traindataset/qtrees_embedding.pkl", "wb") as file:
        pickle.dump(result, file)
    print(time.time() - start_time)
    







