import faiss
import numpy as np
import sqlite3
import pandas as pd
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Optional, Dict, Union
import torch
from sklearn.decomposition import PCA
import pickle

class VectorDatabase:
    def __init__(
        self, 
        index_path: str, 
        dimension: int, 
        metric: str, 
        storage_path: Optional[str],
        column_names: Optional[list[str]],
        encoder_model = None
    ):
        """
        Initializes a FAISS-based vector database.

        :param index_path: Path to the FAISS index file.
        :param dimension: Dimension of stored vectors.
        :param metric: Distance metric ('L2' or 'cosine').
        :param storage_path: Path to store additional metadata (SQLite/Parquet).
        :param model_name: SentenceTransformer model for generating embeddings.
        :param storage_type: 'sqlite' for SQLite, 'parquet' for Parquet file.
        """
        self.index_path = index_path
        self.dimension = dimension
        self.metric = metric
        self.storage_path = storage_path
        self.model = encoder_model
        self.metadata_columns = column_names
        
        # Select FAISS distance metric
        if metric == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Cosine uses Inner Product
        else:
            raise ValueError("Unsupported metric. Use 'L2' or 'cosine'.")
        
        # Load existing FAISS index if available
        try:
            self.index = faiss.read_index(index_path)
        except:
            pass  # If the index file does not exist, it starts fresh
        
        # Storage for metadata
        if storage_path:
            self._init_sqlite()

    def _init_sqlite(self):
        """Initializes SQLite database with user-defined metadata columns."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        # Construct table schema dynamically
        columns_schema = ", ".join([f"{col} TEXT" for col in self.metadata_columns])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {columns_schema}
            )
        """)

        conn.commit()
        conn.close()

    def add_vectors(self, texts: Optional[List[str]] = None, embeddings : Optional[List[List]] = None, metadata: pd.DataFrame = None):
        """
        Adds a batch of text vectors to the FAISS index and stores metadata.

        :param texts: List of text data to be embedded and stored.
        :param extra_data: Optional list of metadata related to each text.
        """
        if embeddings is not None:
            pass
        elif texts is not None and self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        else:
            raise Exception('Either No text/model or embeddings not provided.')

        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            if not embeddings.flags['C_CONTIGUOUS']:
                # Ensure embeddings are float32 and C-contiguous
                embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)  # Save index
        
        # Store metadata
        if self.storage_path and (metadata is not None):
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            placeholders = ", ".join(["?" for _ in self.metadata_columns])
            column_names = ", ".join(self.metadata_columns)

            metadata_values = metadata[self.metadata_columns].fillna("").values.tolist()
            cursor.executemany(f"INSERT INTO metadata ({column_names}) VALUES ({placeholders})", metadata_values)
            conn.commit()
            conn.close()
        print('number of indices: ', self.index.ntotal)

    def search(self, query = None, query_embedding: Optional[List] = None, k: int = 1) -> pd.DataFrame:
        """
        Searches for the top-k most similar texts.

        :param query: The text query to search for.
        :param k: Number of top results to return.
        :return: List of dictionaries containing matching texts and distances.
        """
        # Generate embedding for query
        if query_embedding is not None:
            pass
        elif query is not None and self.model is not None:
            query_embedding = self.model.encode(query).reshape(1, -1).astype("float32")

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding)

        # Perform FAISS search
        distances, indices = self.index.search(query_embedding, k)

        if self.storage_path:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
        else:
            return None
        idx_tuple = tuple(indices[0].tolist())
        idx_tuple = tuple(x + 1 for x in idx_tuple)
        results = []

        if idx_tuple is not None:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM metadata WHERE id IN ({', '.join(['?']*len(idx_tuple))})", idx_tuple)
            fetched_results = cursor.fetchall()
            conn.close()
            # Convert results into a dictionary for quick lookup
            results = pd.DataFrame(fetched_results, columns=["id"] + self.metadata_columns)
            # Construct final result list, maintaining FAISS order
            results['distance']= distances[0].tolist()
            return results
        else:
            return None
        

    def save_index(self):
        """Saves the FAISS index to a file."""
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        """Loads an existing FAISS index from a file."""
        self.index = faiss.read_index(self.index_path)

    def delete_index(self):
        """Deletes the current FAISS index (use with caution)."""
        self.index.reset()
        faiss.write_index(self.index, self.index_path)


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    MODEL = BertModel.from_pretrained('bert-base-uncased').to(device)

    def bertEmbedding(texts, train_pca : bool = True):
        """
        Generate BERT embeddings for a list of text inputs.
        
        :param texts: List of strings (text inputs).
        :return: List of embeddings, one per input text.
        """
        # Tokenize input batch
        inputs = TOKENIZER(texts, return_tensors="pt", padding=True, truncation=True)

        # Move to device (GPU or CPU)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Get BERT embeddings
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = MODEL(**inputs)

        # Extract [CLS] embeddings for each text in the batch
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()

        # Apply PCA to reduce dimensionality
        if train_pca == True:
            target_dimension = 12
            pca = PCA(n_components=target_dimension)
            reduced_embeddings = pca.fit_transform(cls_embeddings)

            with open("models/pca_model_small.pkl", "wb") as f:
                pickle.dump(pca, f)
        else:
            with open("models/pca_model_small.pkl", "rb") as f:
                pca = pickle.load(f)
                reduced_embeddings = pca.transform(cls_embeddings)

        # Convert to list of embeddings
        return reduced_embeddings
    
    df = pd.read_csv('traindataset/Dummy_Queries.csv')
    print(len(df['query']))
    embeddings = bertEmbedding(df['query'].tolist())
    print(len(embeddings))

    vd = VectorDatabase('data/dummy_index.faiss', 12, 'cosine', 'data/dummy_queries.sqlite', ['query'], None)
    #vd.add_vectors(embeddings= embeddings, metadata= df)
    query_embedding= bertEmbedding('SELECT * FROM STUDENTS WHERE id = 45 and age > 18', train_pca= False)
    print(query_embedding)
    results = vd.search(query_embedding=query_embedding, k = 2)
    print(results)
