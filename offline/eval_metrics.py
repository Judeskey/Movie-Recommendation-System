import numpy as np, scipy.sparse as sp

def recall_at_k(preds, truth, k=10):
    hits = 0; total = truth.shape[0]
    truth_items = truth.tocoo()
    truth_by_user = dict(zip(truth_items.row, truth_items.col))
    for u, topk in enumerate(preds):
        t = truth_by_user.get(u, None)
        if t is None:
            total -= 1
            continue
        if t in topk[:k]:
            hits += 1
    return hits / max(total, 1)

def recommend_user(u, U, V, seen_row, k=100, remove_seen=True):
    scores = U[u] @ V.T
    if remove_seen:
        scores[seen_row.indices] = -1e9
    topk = np.argpartition(-scores, k)[:k]
    return topk[np.argsort(-scores[topk])]

def main():
    X_val = sp.load_npz("data/X_val.npz").tocsr()
    X_tr  = sp.load_npz("data/X_train.npz").tocsr()
    als   = np.load("models/als.npz")
    U, V  = als["user_factors"], als["item_factors"]

    preds = [recommend_user(u, U, V, seen_row=X_tr[u], k=50) for u in range(X_val.shape[0])]
    print(f"Recall@10={recall_at_k(preds, X_val, k=10):.4f}")
    print(f"Recall@20={recall_at_k(preds, X_val, k=20):.4f}")

if __name__ == "__main__":
    main()
