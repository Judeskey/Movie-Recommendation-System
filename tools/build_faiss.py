import numpy as np
try:
    import faiss
except Exception as e:
    print("[WARN] FAISS not available; skipping index build. Error:", e)
    raise SystemExit(0)

def main():
    als = np.load("models/als.npz")
    V = als["item_factors"].astype("float32")
    faiss.normalize_L2(V)
    index = faiss.IndexFlatIP(V.shape[1])
    index.add(V)
    faiss.write_index(index, "service/faiss_index.bin")
    print("Wrote service/faiss_index.bin")

if __name__ == "__main__":
    main()
