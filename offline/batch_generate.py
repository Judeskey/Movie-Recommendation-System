import numpy as np, scipy.sparse as sp, json
from pathlib import Path

def main():
    U = np.load("models/als.npz")["user_factors"]
    V = np.load("models/als.npz")["item_factors"]
    X = sp.load_npz("data/X_train.npz").tocsr()
    with open("models/id_mappings.json", encoding="utf-8") as f:
        maps = json.load(f)
    rev_iid = {v: int(k) for k, v in maps["iid_map"].items()}

    K = 50
    Path("data/batch").mkdir(parents=True, exist_ok=True)
    with open("data/batch/user_topk.tsv","w",encoding="utf-8") as w:
        for u in range(U.shape[0]):
            scores = U[u] @ V.T
            scores[X[u].indices] = -1e9
            top = np.argpartition(-scores, K)[:K]
            top = top[np.argsort(-scores[top])]
            items = [str(rev_iid[i]) for i in top]
            w.write(f"{u}\t{','.join(items)}\n")
    print("Wrote data/batch/user_topk.tsv")

if __name__ == "__main__":
    main()
