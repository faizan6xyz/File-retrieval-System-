import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
folder_path = "Data"
all_chunks = []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                all_chunks.append(paragraph)
print(f"Total chunks: {len(all_chunks)}")
model = SentenceTransformer("BAAI/bge-base-en-v1.5") # Load the BGE model for generating embeddings, which captures semantic meaning of text
embeddings = model.encode(all_chunks,convert_to_numpy=True) # Generate embeddings for all text chunks, converting them to a NumPy array for efficient processing and storage. Each chunk is transformed into a high-dimensional vector representation that captures its semantic meaning.
print("Embedding shape:", embeddings.shape)
faiss.normalize_L2(embeddings) # give (10 number of the text chunk , 786 size of each embedding vector )
dimension = embeddings.shape[1] # get the value of dimension
index = faiss.IndexFlatIP(dimension) # this Create an empty FAISS database.
index.add(embeddings.astype(np.float32))# Store all embeddings inside that database for fast similarity search.
print("Vectors stored:", index.ntotal) 
faiss.write_index(index, "rag_index.faiss") # Stores the vector embeddings
np.save("chunks.npy", np.array(all_chunks)) # Stores the original paragraph text corresponding to each vector
print("FAISS index saved successfully.")
# Load FAISS index , index = faiss.read_index("rag_index.faiss") 
# Load text chunks , all_chunks = np.load("chunks.npy", allow_pickle=True)
from rank_bm25 import BM25Okapi
tokenized_chunks = [chunk.lower().split()for chunk in all_chunks]  # create a list of tokenized chunks for BM25
bm25 = BM25Okapi(tokenized_chunks) # Initialize BM25 with the tokenized chunks, allowing for keyword-based search
print("BM25 index created.")
while True:
    query = input("\nEnter query (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    query_embedding = model.encode(query,convert_to_numpy=True)
    query_embedding = np.array([query_embedding],dtype=np.float32)
    faiss.normalize_L2(query_embedding) # Semantic Search (Dense Search) → FAISS + embeddings
    dense_scores, dense_indices = index.search(query_embedding,k=len(all_chunks)) # Semantic Search (Dense Search) → FAISS + embeddings
    dense_scores = dense_scores[0] # Get the scores for the single query for sematic search
    tokenized_query = query.lower().split() # Keyword Search (Sparse Search) → BM25 
    bm25_scores = bm25.get_scores(tokenized_query)  # Keyword Search (Sparse Search) → BM25
    dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)     # Normalize the dense scores 
    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)   # Normalize the BM25 scores
    final_scores = (0.7 * dense_scores_norm + 0.3 * bm25_scores_norm)  # Combine the normalized scores with weights (0.7 for dense and 0.3 for BM25)
    top_indices = np.argsort(final_scores)[::-1][:3] # top 3 results based on the final hybrid score
    print("\nTop matching chunks:\n")
    for rank, idx in enumerate(top_indices):
        print(f"Rank {rank+1}")
        print(f"Hybrid Score: {final_scores[idx]:.4f}")
        print(all_chunks[idx])
        print("-" * 60)