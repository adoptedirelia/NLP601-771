import pickle
import json
import faiss
import numpy as np
from collections import defaultdict

# ---------------------
# 1. 加载 embeddings
# ---------------------

with open("scifact_evidence_embeddings.pkl", "rb") as f:
    abstract_embeddings = pickle.load(f)  # {doc_id: np.array}

with open("scifact_claim_embeddings.pkl", "rb") as f:
    claim_embeddings = pickle.load(f)  # {claim_id: np.array}

# with open("ground_truth.json", "r") as f:
#     ground_truth = json.load(f)  # {claim_id: correct_doc_id}



# ---------------------
# 2. 构建 FAISS Index
# ---------------------

doc_ids = list(abstract_embeddings.keys())
doc_vectors = np.stack([abstract_embeddings[doc_id] for doc_id in doc_ids]).astype("float32")

faiss.normalize_L2(doc_vectors)  # cosine similarity
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_vectors)

# ---------------------
# 3. 查询并评估
# ---------------------

total = 0
correct = 0

for claim_id, query_vec in claim_embeddings.items():
    query = query_vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)
    D, I = index.search(query, k=1)
    retrieved_doc_id = doc_ids[I[0][0]]

    gt_doc_id = ground_truth.get(claim_id)
    if gt_doc_id is not None:
        total += 1
        if retrieved_doc_id == gt_doc_id:
            correct += 1

recall_at_1 = correct / total if total > 0 else 0.0

# ---------------------
# 4. 打印结果
# ---------------------

print(f"Recall@1: {recall_at_1:.4f}")

