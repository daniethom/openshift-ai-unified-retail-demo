import os
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# --- Configuration from Environment Variables ---
MILVUS_URI = os.getenv("MILVUS_URI")
COLLECTION_NAME = "product_data"
MODEL_NAME = 'all-MiniLM-L6-v2'

def prime_database():
    print("Starting database priming...")
    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} not found. Exiting.")
        return

    model = SentenceTransformer(MODEL_NAME)
    data = pd.read_csv("data/retail_products.csv")

    # --- Generate Embeddings and Insert Data (add your specific logic) ---
    print(f"Processing {len(data)} records...")
    # for index, row in data.iterrows():
    #     # ... your embedding and insertion logic here ...

    print("Database priming complete.")

if __name__ == "__main__":
    prime_database()