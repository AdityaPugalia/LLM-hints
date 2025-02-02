from transformers import BertTokenizer, BertModel
import torch
from qep import Graph
import json
from db import Database
import config
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained('bert-base-uncased').to(device)

def bertEmbedding(text):
    
    # Tokenize input
    inputs = TOKENIZER(text, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get BERT embeddings
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = MODEL(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Singular value embedding
    return cls_embedding.squeeze().tolist()

def getCondition(node):
     if node['Index Condition'] is not None:
          return bertEmbedding(node['Index Condition'])
     elif node['Join Condition'] is not None:
          return bertEmbedding(node['Join Condition']) 
     else: 
          return bertEmbedding('None')
     
def extractConfigurations(file_path):
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
     
def createGraphFeatures(nodes):
    
    config = extractConfigurations('data/config_settings.json')
    
    
    shared_buffers_inv = 1 / config['shared_buffers']
    work_mem_inv = 1 / config['work_mem']
    hash_mem_multiplier_inv = 1 / (config['work_mem'] * config['hash_mem_multiplier'])

    # Extract node data into separate arrays
    labels = np.array([node['label'] for node in nodes])
    cumulative_costs = np.array([node['Cummulative Cost'] for node in nodes])
    planned_rows = np.array([node['Planned rows'] for node in nodes])
    widths = np.array([node['Width'] for node in nodes])
    filters = np.array([node['Filter'] for node in nodes])
    parent_ids = np.array([node['parent id'] for node in nodes])

    # Step 1: Calculate label embeddings in a vectorized way
    label_embeddings = np.array([bertEmbedding(label) for label in labels])

    # Step 2: Calculate planned row ratios (shared_buffers)
    planned_row_ratios = planned_rows * widths * shared_buffers_inv

    # Step 3: Compute condition features
    condition_features = np.array([getCondition(node) for node in nodes])

    # Step 4: Calculate filter embeddings (0 for None, otherwise apply bertEmbedding)
    filter_embeddings = np.array([
        bertEmbedding('No Filter') if filter_value is None else bertEmbedding(filter_value) 
        for filter_value in filters
    ])

    # Step 5: Work memory ratios (handle hash join condition)
    work_mem_ratios = np.zeros(len(nodes))
    for i, parent_id in enumerate(parent_ids):
        if parent_id is not None:
            parent_label = nodes[parent_id]['label']
            if parent_label == 'hash join':
                work_mem_ratios[i] = planned_rows[i] * widths[i] * hash_mem_multiplier_inv
            else:
                work_mem_ratios[i] = planned_rows[i] * widths[i] * work_mem_inv

    # Step 6: Combine all features into a single array
    features = np.column_stack((
        label_embeddings,  # Assume each embedding is a vector
        cumulative_costs,
        planned_row_ratios,
        condition_features,
        filter_embeddings,
        work_mem_ratios
    ))

    return features

if __name__ == '__main__':
    db = Database(user= config.USER, dbname= config.DBASE)
    db.connect()
    command = 'SELECT COUNT(*co) FROM customer INNER JOIN orders ON customer.c_custkey = orders.o_custkey WHERE o_orderdate > \'2015-01-01\';'





