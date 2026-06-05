"""
collab_recommender.py
----------------------
Collaborative Filtering using two algorithms from the Surprise library:

  1. SVD  (Singular Value Decomposition)
     - Matrix factorization that discovers hidden latent factors
     - Most accurate for rating prediction
     - e.g. hidden factors like "dark psychological films" or "feel-good comedies"

  2. KNN  (K-Nearest Neighbors) — Item-based
     - Finds K most similar movies based on user rating patterns
     - Very explainable: "X users who loved Movie A also loved Movie B"
     - Uses Cosine similarity as distance metric

Both models are trained, and their predictions are blended in the
HybridRecommender for best results.
"""

import numpy as np
import pandas as pd
import joblib
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# We handle the import gracefully so the rest of the app doesn't crash
# if surprise isn't installed yet
try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_OK = True
except ImportError:
    SURPRISE_OK = False
    print("⚠️  scikit-surprise not installed. Run: pip install scikit-surprise")


class CollabRecommender:
    def __init__(self, k_neighbors: int = 20):
        self.k              = k_neighbors
        self.svd_model      = None
        self.knn_model      = None   # sklearn NearestNeighbors (sparse)
        self.trainset       = None
        self.movies         = None
        self.ratings        = None
        self.surprise_data  = None
        # KNN sparse index structures
        self._knn_matrix    = None   # csr_matrix (items × users)
        self._knn_movieIds  = None   # array mapping row index → movieId
        self._mid_to_row    = None   # dict movieId → row index

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """
        Train both SVD and KNN models on the ratings data.

        Parameters
        ----------
        movies  : movies DataFrame (movieId, title, genres, ...)
        ratings : ratings DataFrame (userId, movieId, rating, timestamp)
        """
        if not SURPRISE_OK:
            print("❌ Cannot train — scikit-surprise not available")
            return self

        self.movies  = movies.reset_index(drop=True)
        self.ratings = ratings

        print("📊 Preparing Surprise dataset...")
        reader = Reader(rating_scale=(1, 5))
        data   = Dataset.load_from_df(
            ratings[["userId", "movieId", "rating"]], reader
        )
        self.trainset      = data.build_full_trainset()
        self.surprise_data = data

        # ── SVD ──────────────────────────────────────────────────────
        print("🔢 Training SVD model...")
        self.svd_model = SVD(
            n_factors=150,      # latent factors — more = captures subtler patterns
            n_epochs=20,        # training iterations
            lr_all=0.005,       # learning rate
            reg_all=0.02,       # regularization to prevent overfitting
            random_state=42
        )
        self.svd_model.fit(self.trainset)
        print("✅ SVD model trained")

        # ── KNN (Item-based, sparse sklearn) ─────────────────────────
        # Filter to items with ≥50 ratings so the matrix stays small.
        # 39k items × 39k items @ int64 = 11 GiB — way too large.
        # With sparse NearestNeighbors we only store non-zero entries.
        print(f"🔗 Training KNN model (k={self.k}, item-based, cosine, sparse)...")
        MIN_RATINGS = 50
        item_counts = ratings["movieId"].value_counts()
        popular_ids = item_counts[item_counts >= MIN_RATINGS].index
        knn_ratings = ratings[ratings["movieId"].isin(popular_ids)]
        print(f"   Items after min-{MIN_RATINGS}-rating filter: "
              f"{len(popular_ids):,} (was {item_counts.shape[0]:,})")

        # Build pivot: rows=movies, cols=users, values=rating (0 where unrated)
        pivot = knn_ratings.pivot_table(
            index="movieId", columns="userId", values="rating", fill_value=0
        )
        self._knn_movieIds = np.array(pivot.index)
        self._mid_to_row   = {mid: i for i, mid in enumerate(self._knn_movieIds)}
        self._knn_matrix   = csr_matrix(pivot.values)

        self.knn_model = NearestNeighbors(
            n_neighbors=self.k + 1,   # +1 because the item itself is returned
            metric="cosine",
            algorithm="brute",        # required for sparse cosine
            n_jobs=-1
        )
        self.knn_model.fit(self._knn_matrix)
        print("✅ KNN model trained (sparse)")

        return self

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------

    def recommend_svd(
        self,
        user_id: int,
        top_n: int = 10,
        seen_movies: list = None
    ) -> pd.DataFrame:
        """
        SVD: Predict ratings for all unrated movies for a given user,
        return top_n highest predicted.
        """
        if self.svd_model is None or self.movies is None:
            return pd.DataFrame()

        seen = set(seen_movies or [])
        rated_ids = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        )

        all_movie_ids = self.movies["movieId"].tolist()
        predictions   = []

        for mid in all_movie_ids:
            title = self.movies[self.movies["movieId"] == mid]["title"].values
            if len(title) == 0:
                continue
            title = title[0]
            if mid in rated_ids or title in seen:
                continue
            pred = self.svd_model.predict(user_id, mid)
            predictions.append({
                "title"          : title,
                "genres"         : self.movies[self.movies["movieId"] == mid]["genres"].values[0],
                "year"           : self.movies[self.movies["movieId"] == mid]["year"].values[0],
                "avg_rating"     : self.movies[self.movies["movieId"] == mid]["avg_rating"].values[0],
                "rating_count"   : self.movies[self.movies["movieId"] == mid]["rating_count"].values[0],
                "predicted_rating": round(pred.est, 3),
                "source"         : "Collaborative (SVD)"
            })

        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return pd.DataFrame(predictions[:top_n])

    def recommend_knn(
        self,
        movie_title: str,
        top_n: int = 10,
        seen_movies: list = None
    ) -> pd.DataFrame:
        """
        KNN: Find K nearest neighbor movies based on rating patterns.
        Uses sparse sklearn NearestNeighbors — memory-safe for large datasets.
        """
        if self.knn_model is None or self.movies is None:
            return pd.DataFrame()

        movie_row = self.movies[self.movies["title"] == movie_title]
        if movie_row.empty:
            return pd.DataFrame()

        mid = movie_row["movieId"].values[0]
        row_idx = self._mid_to_row.get(mid)
        if row_idx is None:
            return pd.DataFrame()   # movie not in sparse KNN index

        # Query the sparse model — returns distances + indices
        query_vec = self._knn_matrix[row_idx]   # shape (1, n_users)
        distances, indices = self.knn_model.kneighbors(
            query_vec, n_neighbors=min(top_n * 3 + 1, len(self._knn_movieIds))
        )

        seen = set(seen_movies or [])
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            neighbor_mid = self._knn_movieIds[idx]
            if neighbor_mid == mid:
                continue   # skip the query item itself
            row = self.movies[self.movies["movieId"] == neighbor_mid]
            if row.empty:
                continue
            title = row["title"].values[0]
            if title in seen:
                continue

            sim = round(1.0 - float(dist), 4)   # cosine distance → similarity
            results.append({
                "title"        : title,
                "genres"       : row["genres"].values[0],
                "year"         : row["year"].values[0],
                "avg_rating"   : row["avg_rating"].values[0],
                "rating_count" : row["rating_count"].values[0],
                "similarity"   : sim,
                "source"       : "Collaborative (KNN)"
            })
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def explain_knn(self, movie_title: str, top_n: int = 5) -> dict:
        """
        KNN Explanation: similarity scores for the nearest neighbors.
        Returns neighbor details for the 'Why This?' feature.
        """
        recs = self.recommend_knn(movie_title, top_n=top_n)
        if recs.empty:
            return {}
        details = recs[["title", "similarity"]].to_dict(orient="records")
        return {
            "movie"    : movie_title,
            "neighbors": details,
            "k"        : self.k
        }

    # ------------------------------------------------------------------
    # PERSIST
    # ------------------------------------------------------------------

    def save(self, path: str = "models/collab_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 Collab model saved → {path}")

    @staticmethod
    def load(path: str = "models/collab_model.pkl"):
        model = joblib.load(path)
        print(f"📦 Collab model loaded ← {path}")
        return model
