import os, json
from typing import Optional, List

import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# ---------- Load model artifacts ----------
ALS = np.load("models/als.npz")
U = ALS["user_factors"].astype(np.float32)   # shape: [n_users, d]
V = ALS["item_factors"].astype(np.float32)   # shape: [n_items, d]
X_train = sp.load_npz("data/X_train.npz").tocsr()  # USER x ITEM

# ---------- ID mappings ----------
with open("models/id_mappings.json", encoding="utf-8") as f:
    maps = json.load(f)
uid_map = {int(k): int(v) for k, v in maps["uid_map"].items()}  # raw userId -> row index
iid_map = {int(k): int(v) for k, v in maps["iid_map"].items()}  # raw movieId -> col index
rev_iid = {v: k for k, v in iid_map.items()}                    # col index -> raw movieId

# ---------- Pretty metadata (safe if CSV missing) ----------
try:
    ITEMS = pd.read_csv("data/items_meta.csv")  # columns: movieId,title,genres
    id2meta = {int(r.movieId): {"title": r.title, "genres": r.genres}
               for _, r in ITEMS.iterrows()}
except Exception:
    id2meta = {}  # fall back to empty meta

def enrich(items: List[int]) -> List[dict]:
    """Return list of dicts with movieId + optional title/genres."""
    return [{"movieId": int(mid), **id2meta.get(int(mid), {})} for mid in items]

# ---------- Optional FAISS index for similar() ----------
try:
    import faiss
    INDEX = faiss.read_index("service/faiss_index.bin")
    use_faiss = True
except Exception:
    INDEX = None
    use_faiss = False
# ---------- FastAPI ----------
app = FastAPI(title="Movie Recs API (ML-1M)", debug=True)

class RecRequest(BaseModel):
    user_id: int
    k: int = 10

class SimilarRequest(BaseModel):
    item_id: int
    k: int = 10

class Feedback(BaseModel):
    user_id: int
    item_id: int
    event: str = "view"  # view/click/like/purchase
    weight: float = 1.0
    timestamp: Optional[int] = None

# ---------- Core helper ----------
def _recommend_for_user(u_idx: int, k: int) -> List[int]:
    """Return top-k raw movieIds for an internal user index."""
    if u_idx < 0 or u_idx >= U.shape[0]:
        raise HTTPException(status_code=404, detail="Unknown user")
    scores = U[u_idx] @ V.T
    # filter seen items
    seen = X_train[u_idx].indices
    scores[seen] = -1e9
    topk_idx = np.argpartition(-scores, k)[:k]
    topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
    return [int(rev_iid[i]) for i in topk_idx]
# ---------- Core helper ----------

# ✅ PASTE THIS BLOCK HERE
@app.get("/debug/user/{user_id}")
def debug_user(user_id: int, k: int = 10):
    if user_id not in uid_map:
        return {"known": False, "reason": "cold-start user; falling back to popularity"}
    u_idx = uid_map[user_id]
    seen_idx = X_train[u_idx].indices
    seen = [int(rev_iid[i]) for i in seen_idx]
    recs = _recommend_for_user(u_idx, k)
    overlap = sorted(set(seen) & set(recs))
    return {
        "known": True,
        "u_idx": int(u_idx),
        "seen_count": int(len(seen_idx)),
        "sample_seen_meta": enrich(seen[:5]),
        "recs": recs,
        "recs_meta": enrich(recs),
        "overlap_with_seen": overlap  # should be []
    }
# ---------- Endpoints ----------
@app.get("/healthz")
def health():
    return {"status": "ok", "users": int(U.shape[0]), "items": int(V.shape[0]), "faiss": use_faiss}

@app.post("/recommend")
def recommend(req: RecRequest):
    # Cold-start: popularity
    if req.user_id not in uid_map:
        pop = np.asarray(X_train.sum(axis=0)).ravel()
        top = np.argpartition(-pop, req.k)[:req.k]
        top = top[np.argsort(-pop[top])]
        items = [int(rev_iid[i]) for i in top.tolist()]
        return {"user_id": req.user_id, "items": items, "items_meta": enrich(items)}

    # Known user → personalized
    u_idx = uid_map[req.user_id]
    items = _recommend_for_user(u_idx, req.k)
    return {"user_id": req.user_id, "items": items, "items_meta": enrich(items)}

@app.post("/similar")
def similar(req: SimilarRequest):
    if req.item_id not in iid_map:
        raise HTTPException(status_code=404, detail="Unknown item")
    i_idx = iid_map[req.item_id]

    if use_faiss:
        vec = V[i_idx:i_idx+1].astype("float32")
        faiss.normalize_L2(vec)
        _, I = INDEX.search(vec, req.k + 1)
        nn_idx = [int(j) for j in I[0] if int(j) != int(i_idx)][:req.k]
        nn = [int(rev_iid[j]) for j in nn_idx]
    else:
        sims = V @ V[i_idx]
        sims[i_idx] = -1e9
        top = np.argpartition(-sims, req.k)[:req.k]
        top = top[np.argsort(-sims[top])]
        nn = [int(rev_iid[i]) for i in top]

    return {"item_id": req.item_id, "items": nn, "items_meta": enrich(nn)}

@app.post("/feedback")
def feedback(fb: Feedback):
    os.makedirs("data/events", exist_ok=True)
    with open("data/events/interactions.jsonl", "a", encoding="utf-8") as w:
        w.write(json.dumps(fb.dict()) + "\n")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)