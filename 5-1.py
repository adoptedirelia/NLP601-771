import pickle
import numpy as np
import faiss
from datasets import load_dataset


class FaissIRSystem:
    def __init__(self, abstract_embedding_path, claim_embedding_path):
        self.abstract_embedding_path = abstract_embedding_path
        self.claim_embedding_path = claim_embedding_path

        self.abstract_embeddings = self._load_embeddings(self.abstract_embedding_path)
        self.claim_embeddings = self._load_embeddings(self.claim_embedding_path)

        self.embedding_dim = len(next(iter(self.abstract_embeddings.values())))
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        self.doc_id_map = []
        self._build_index()

    def _load_embeddings(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build_index(self):
        embedding_matrix = []

        for doc, embedding in self.abstract_embeddings.items():
            doc_id, abstract_text = doc
            self.doc_id_map.append((str(doc_id), abstract_text))
            embedding_matrix.append(np.array(embedding).astype("float32"))

        embedding_matrix = np.vstack(embedding_matrix).astype("float32")
        self.index.add(embedding_matrix)

    def retrieve(self, claim_embedding, top_k=5):
        query_vector = np.array([claim_embedding]).astype("float32")
        D, I = self.index.search(query_vector, top_k)

        results = []
        for idx in I[0]:
            doc_id, abstract_text = self.doc_id_map[idx]
            results.append((doc_id, abstract_text))
        return results

    def retrieve_all_claims(self, top_k=5, limit=None):
        results_all = {}
        for i, (claim_doc, claim_embedding) in enumerate(self.claim_embeddings.items()):
            claim_id, claim_text = claim_doc
            results = self.retrieve(claim_embedding, top_k=top_k)
            results_all[str(claim_id)] = {
                "claim": claim_text,
                "results": results
            }
            if limit is not None and i + 1 >= limit:
                break
        return results_all


    def evaluate(self, ground_truth: dict, top_k=10):
        mrr_total = 0.0
        map_total = 0.0
        count = 0

        for claim_doc, claim_embedding in self.claim_embeddings.items():
            claim_id, _ = claim_doc
            claim_id_str = str(claim_id)
            relevant_docs = ground_truth.get(claim_id_str, [])
            if not relevant_docs:
                continue 

            results = self.retrieve(claim_embedding, top_k=top_k)
            retrieved_ids = [doc_id for doc_id, _ in results]

            rr = 0.0
            for rank, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_total += rr

            num_relevant = 0
            ap = 0.0
            for rank, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_docs:
                    num_relevant += 1
                    ap += num_relevant / (rank + 1)
            if num_relevant > 0:
                ap /= len(relevant_docs)
            map_total += ap

            count += 1

        mean_mrr = mrr_total / count if count > 0 else 0
        mean_ap = map_total / count if count > 0 else 0

        print(f"MRR: {mean_mrr:.4f}")
        print(f"MAP: {mean_ap:.4f}")


ir_system = FaissIRSystem(
    abstract_embedding_path="scifact_evidence_embeddings.pkl",
    claim_embedding_path="scifact_claim_embeddings.pkl"
)


dataset = load_dataset("allenai/scifact", "claims")
ground_truth = {}

for item in dataset["train"]:

    claim_id = str(item["id"])
    relevant_doc_ids = item['cited_doc_ids']  # list of integers
    ground_truth[claim_id] = [str(doc_id) for doc_id in relevant_doc_ids]

ir_system.evaluate(ground_truth, top_k=1)
ir_system.evaluate(ground_truth, top_k=10)
ir_system.evaluate(ground_truth, top_k=50)
