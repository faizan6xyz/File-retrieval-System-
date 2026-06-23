import os
import numpy as np
import faiss
import pickle
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
os.environ["FASTEMBED_CACHE_PATH"] = r"C:\Users\faiza\fastembed_models"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE_TOKENS = 400  # Safe limit for BGE (max 512)
OVERLAP_TOKENS = 50
BATCH_SIZE = 32  # Process embeddings in batches to save RAM
_model = TextEmbedding(MODEL_NAME)
def chunk_text_smart(text, chunk_size=CHUNK_SIZE_TOKENS, overlap=OVERLAP_TOKENS):
    words = text.split()
    chunks = []
    start = 0
    overlap_words = max(1, int(overlap / 1.3))
    while start < len(words):
        current_chunk_words = words[start:]
        end_offset = 0
        current_token_count = 0
        for i, word in enumerate(current_chunk_words):
            # Rough token count per word (1 + length/4 is a common heuristic)
            token_est = 1 + len(word) / 4
            if current_token_count + token_est > chunk_size:
                break
            current_token_count += token_est
            end_offset = i + 1
        if end_offset == 0:
            end_offset = 1  # Prevent infinite loop on a single huge word
        chunk = " ".join(words[start:start + end_offset])
        chunks.append(chunk)
        advance = max(1, end_offset - overlap_words)
        start += advance
    return chunks
def load_existing_data(chunk_path, index_path):
    full_index_path = os.path.join(chunk_path, index_path)
    full_chunks_path = os.path.join(chunk_path, "chunks.npy")
    full_meta_path = os.path.join(chunk_path, "metadata.npy")
    index = None
    existing_chunks = []
    existing_metadata = []
    dimension = 768 
    if os.path.exists(full_index_path):
        index = faiss.read_index(full_index_path)
        dimension = index.d
        existing_chunks = np.load(full_chunks_path, allow_pickle=True).tolist()
        if os.path.exists(full_meta_path):
            existing_metadata = np.load(full_meta_path, allow_pickle=True).tolist()
        print(f"Loaded existing index with {index.ntotal} vectors.")
    else:
        print("No existing index found. Creating new one.")
    return index, existing_chunks, existing_metadata, dimension
def save_data(index, chunks, metadata, chunk_path, index_path, bm25):
    full_index_path = os.path.join(chunk_path, index_path)
    full_chunks_path = os.path.join(chunk_path, "chunks.npy")
    full_meta_path = os.path.join(chunk_path, "metadata.npy")
    full_bm25_path = os.path.join(chunk_path, "bm25.pkl")
    faiss.write_index(index, full_index_path + ".tmp")
    os.replace(full_index_path + ".tmp", full_index_path)
    tmp_chunks_path = full_chunks_path.replace(".npy", ".tmp.npy")
    tmp_meta_path = full_meta_path.replace(".npy", ".tmp.npy")
    np.save(tmp_chunks_path, np.array(chunks, dtype=object))
    np.save(tmp_meta_path, np.array(metadata, dtype=object))
    os.replace(tmp_chunks_path, full_chunks_path)
    os.replace(tmp_meta_path, full_meta_path)
    with open(full_bm25_path + ".tmp", 'wb') as f:
        pickle.dump(bm25, f)
    os.replace(full_bm25_path + ".tmp", full_bm25_path)
    print("All data saved successfully.")
def build_index(file_name,
                 folder_path="SYSTEM/Data",
                 chunk_path="SYSTEM/RAG_data",
                 index_path="rag_index.faiss"):
    os.makedirs(chunk_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    index, existing_chunks, existing_metadata, dimension = load_existing_data(
        chunk_path, index_path
    )
    if index is None:
        index = faiss.IndexFlatIP(dimension)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    new_chunks = chunk_text_smart(text)
    print(f"Generated {len(new_chunks)} new chunks from {file_name}")
    existing_set = set(existing_chunks)
    unique_new_chunks = []
    unique_new_metadata = []
    for chunk in new_chunks:
        if chunk not in existing_set:
            chunk_id = len(existing_chunks) + len(unique_new_chunks)
            unique_new_chunks.append(chunk)
            unique_new_metadata.append({
                "source": file_name,
                "chunk_id": chunk_id
            })
    if not unique_new_chunks:
        print("No new unique chunks to add.")
        return
    print(f"Adding {len(unique_new_chunks)} unique chunks after deduplication.")
    all_embeddings = []
    for i in range(0, len(unique_new_chunks), BATCH_SIZE):
        batch = unique_new_chunks[i:i + BATCH_SIZE]
        batch_embs = np.stack(list(_model.embed(batch))).astype(np.float32)
        faiss.normalize_L2(batch_embs)
        all_embeddings.append(batch_embs)
    if not all_embeddings:
        return
    new_embeddings = np.vstack(all_embeddings)
    index.add(new_embeddings)
    final_chunks = existing_chunks + unique_new_chunks
    final_metadata = existing_metadata + unique_new_metadata
    print("Rebuilding BM25 index...")
    tokenized_corpus = [chunk.lower().split() for chunk in final_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    save_data(index, final_chunks, final_metadata, chunk_path, index_path, bm25)
    print(f"Indexing complete. Total vectors: {index.ntotal}")
if __name__ == "__main__":
    try:
        build_index("testing.txt")
    except Exception as e:
        print(f"Error during indexing: {e}")