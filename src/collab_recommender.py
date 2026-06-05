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

# We handle the import gracefully so the rest of the app doesn't crash
# if surprise isn't installed yet
try:
    from surprise import SVD, KNNWithMeans, Dataset, Reader
    from surprise.model_selection import cross_validate
    SURPRISE_OK = True
except ImportError:
    SURPRISE_OK = False
    print("⚠️  scikit-surprise not installed. Run: pip install scikit-surprise")


class CollabRecommender:
    def __init__(self, k_neighbors: int = 40):
        self.k           = k_neighbors
        self.svd_model   = None
        self.knn_model   = None
        self.trainset    = None
        self.movies      = None
        self.ratings     = None
        self.surprise_data = None

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

        # ── KNN (Item-based) ─────────────────────────────────────────
        print(f"🔗 Training KNN model (k={self.k}, item-based, cosine)...")
        knn_options = {
            "name"     : "cosine",   # distance metric
            "user_based": False      # item-based (movie-to-movie)
        }
        self.knn_model = KNNWithMeans(
            k=self.k,
            sim_options=knn_options,
            verbose=False
        )
        self.knn_model.fit(self.trainset)
        print("✅ KNN model trained")

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
        Returns movies most similar to the given movie.
        """
        if self.knn_model is None or self.movies is None:
            return pd.DataFrame()

        # Map title → internal Surprise iid
        movie_row = self.movies[self.movies["title"] == movie_title]
        if movie_row.empty:
            return pd.DataFrame()

        mid = movie_row["movieId"].values[0]
        try:
            inner_id  = self.trainset.to_inner_iid(mid)
        except ValueError:
            return pd.DataFrame()

        # Get K nearest neighbors
        neighbors = self.knn_model.get_neighbors(inner_id, k=top_n * 3)
        seen = set(seen_movies or [])

        results = []
        for neighbor_inner in neighbors:
            neighbor_mid = self.trainset.to_raw_iid(neighbor_inner)
            row = self.movies[self.movies["movieId"] == neighbor_mid]
            if row.empty:
                continue
            title = row["title"].values[0]
            if title == movie_title or title in seen:
                continue

            # KNN similarity score
            sim = float(self.knn_model.sim[inner_id][neighbor_inner])

            results.append({
                "title"        : title,
                "genres"       : row["genres"].values[0],
                "year"         : row["year"].values[0],
                "avg_rating"   : row["avg_rating"].values[0],
                "rating_count" : row["rating_count"].values[0],
                "similarity"   : round(sim, 4),
                "source"       : "Collaborative (KNN)"
            })
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def explain_knn(self, movie_title: str, top_n: int = 5) -> dict:
        """
        KNN Explanation: how many shared-raters and what's the similarity score.
        Returns neighbor details for the 'Why This?' feature.
        """
        if self.knn_model is None:
            return {}

        movie_row = self.movies[self.movies["title"] == movie_title]
        if movie_row.empty:
            return {}

        mid = movie_row["movieId"].values[0]
        try:
            inner_id  = self.trainset.to_inner_iid(mid)
        except ValueError:
            return {}

        neighbors = self.knn_model.get_neighbors(inner_id, k=top_n)
        details   = []
        for n in neighbors:
            n_mid  = self.trainset.to_raw_iid(n)
            n_row  = self.movies[self.movies["movieId"] == n_mid]
            if n_row.empty:
                continue
            sim    = float(self.knn_model.sim[inner_id][n])
            details.append({
                "title"     : n_row["title"].values[0],
                "similarity": round(sim, 4)
            })

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
