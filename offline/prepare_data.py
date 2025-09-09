import json, pandas as pd, numpy as np, scipy.sparse as sp
from pathlib import Path
import sys

RAW_DIR = Path("data/movielens/ml-1m")
OUT_DIR = Path("data")
MODELS_DIR = Path("models")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def ensure_files():
    needed = ["ratings.dat", "movies.dat"]
    missing = [f for f in needed if not (RAW_DIR / f).exists()]
    if missing:
        print(f"[ERROR] Missing files in {RAW_DIR}: {missing}", file=sys.stderr)
        print("Download ml-1m.zip and extract so you have ratings.dat, movies.dat, users.dat", file=sys.stderr)
        sys.exit(1)

def main():
    ensure_files()

    # 1) Load ML-1M .dat files (UserID::MovieID::Rating::Timestamp)
    ratings = pd.read_csv(
        RAW_DIR / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        RAW_DIR / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

    # 2) Implicit weights: >= 4.0 => positive
    ratings["implicit"] = (ratings["rating"] >= 4.0).astype(np.float32)

    # 3) Map raw ids -> contiguous indices (ensure Python int keys)
    u_unique = ratings["userId"].astype(int).unique().tolist()
    i_unique = ratings["movieId"].astype(int).unique().tolist()

    uid_map = {int(u): int(i) for i, u in enumerate(u_unique)}
    iid_map = {int(m): int(i) for i, m in enumerate(i_unique)}

    ratings["uidx"] = ratings["userId"].astype(int).map(uid_map)
    ratings["iidx"] = ratings["movieId"].astype(int).map(iid_map)

    # 4) Time-aware split: last interaction per user as validation
    ratings = ratings.sort_values(["userId", "timestamp"])
    def split_user(g, holdout=1):
        if len(g) <= holdout:
            return g, g.iloc[0:0]
        return g.iloc[:-holdout], g.iloc[-holdout:]

    train_parts, val_parts = [], []
    for _, g in ratings.groupby("uidx", sort=False):
        tr, va = split_user(g, holdout=1)
        train_parts.append(tr)
        if len(va):
            val_parts.append(va)

    train = pd.concat(train_parts, ignore_index=True)
    val   = pd.concat(val_parts,   ignore_index=True)

    # 5) Sparse matrices
    n_users = ratings["uidx"].nunique()
    n_items = ratings["iidx"].nunique()

    def to_csr(df, col="implicit"):
        return sp.csr_matrix(
            (df[col].values, (df["uidx"].values, df["iidx"].values)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

    X_train = to_csr(train, "implicit")
    X_val   = to_csr(val,   "implicit")

    sp.save_npz(OUT_DIR / "X_train.npz", X_train)
    sp.save_npz(OUT_DIR / "X_val.npz",   X_val)

    # Save item meta for debugging
    movies[["movieId", "title", "genres"]].to_csv(OUT_DIR / "items_meta.csv", index=False)

    # Save id mappings for API
    with open(MODELS_DIR / "id_mappings.json", "w", encoding="utf-8") as f:
        json.dump({"uid_map": uid_map, "iid_map": iid_map}, f)

    print("Prepared ML-1M")
    print("  Train shape:", X_train.shape, " Val shape:", X_val.shape)
    print("  Users:", n_users, " Items:", n_items)

if __name__ == "__main__":
    main()
