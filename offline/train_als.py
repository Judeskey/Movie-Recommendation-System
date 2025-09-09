import numpy as np, scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from pathlib import Path

def main():
    # Load USER×ITEM matrix (rows = users, cols = items)
    X = sp.load_npz("data/X_train.npz").tocsr()

    # Hyperparameters
    factors = 128
    regularization = 0.08
    alpha = 20.0
    iterations = 20

    # Confidence: C = 1 + alpha * r  (keep orientation as USER×ITEM)
    Xui = X.copy()
    Xui.data = 1.0 + alpha * Xui.data

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        calculate_training_loss=True
    )
    model.fit(Xui)

    Path("models").mkdir(exist_ok=True)
    np.savez("models/als.npz",
             user_factors=model.user_factors.astype("float32"),  # shape ≈ (6040, 128)
             item_factors=model.item_factors.astype("float32"))  # shape ≈ (3706, 128)
    print("Saved models/als.npz")

if __name__ == "__main__":
    main()
