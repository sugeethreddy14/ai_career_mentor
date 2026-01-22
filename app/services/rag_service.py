import faiss
import numpy as np

class RAGService:
    def __init__(self, vector_db_path: str):
        try:
            self.index = faiss.read_index(vector_db_path)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
            self.index = None

    def get_similar_profiles(self, user_embedding: np.ndarray, top_k=3) -> list:
        if self.index is None:
            return []
        distances, ids = self.index.search(user_embedding, top_k)
        return [f"Profile_{idx}" for idx in ids[0]]
