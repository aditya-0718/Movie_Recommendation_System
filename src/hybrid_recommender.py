"""
hybrid_recommender.py
----------------------
Hybrid Recommender System — blends all three models:

  Content Score  (TF-IDF / BERT)     weight = 0.30
  SVD Score      (predicted rating)   weight = 0.40
  KNN Score      (rating pattern sim) weight = 0.30

Final score = weighted sum → ranked → top N returned

Also implements:
  - Mood → Genre mapping
  - Era / Decade filtering
  - Seen-It filter
  - Full "Why This?" explanation
  - Taste Profile personalization
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Mood → Genre Mapping ────────────────────────────────────────────────────
MOOD_GENRE_MAP = {
    "😢 Sad / Need Comfort"    : ["Comedy", "Animation", "Family", "Musical"],
    "😤 Stressed / Overwhelmed": ["Comedy", "Animation", "Romance", "Musical"],
    "🤩 Adventurous"           : ["Action", "Adventure", "Sci-Fi", "Fantasy"],
    "😴 Lazy Sunday"           : ["Drama", "Romance", "Comedy"],
    "😱 Want Thrills"          : ["Thriller", "Horror", "Mystery", "Crime"],
    "🧠 Want to Think"         : ["Sci-Fi", "Drama", "Mystery", "Documentary"],
    "💕 Romantic Mood"         : ["Romance", "Drama"],
    "🎉 Party / Fun"           : ["Comedy", "Animation", "Musical", "Action"],
    "No Mood Filter"           : []
}


class HybridRecommender:
    def __init__(
        self,
        content_model=None,
        collab_model=None,
        w_content: float = 0.30,
        w_svd    : float = 0.40,
        w_knn    : float = 0.30
    ):
        self.content  = content_model
        self.collab   = collab_model
        self.w_content = w_content
        self.w_svd     = w_svd
        self.w_knn     = w_knn
        # User session state
        self.seen_movies    = []         # watched list
        self.taste_profile  = {}         # title → rating (1-5)

    # ------------------------------------------------------------------
    # CORE RECOMMENDATION
    # ------------------------------------------------------------------

    def recommend(
        self,
        movie_title  : str,
        user_id      : Optional[int] = None,
        top_n        : int  = 10,
        mood         : str  = "No Mood Filter",
        decade_filter: Optional[int] = None,
        use_hybrid   : bool = True
    ) -> pd.DataFrame:
        """
        Main recommendation entry point.

        Parameters
        ----------
        movie_title   : movie the user searched for
        user_id       : if provided, SVD personalises to this user
        top_n         : how many results to return
        mood          : mood string from MOOD_GENRE_MAP keys
        decade_filter : e.g. 1990 for 1990s only
        use_hybrid    : if False, returns content-only results
        """
        mood_genres = MOOD_GENRE_MAP.get(mood, [])

        # ── Content-Based ────────────────────────────────────────────
        content_df = pd.DataFrame()
        if self.content is not None:
            content_df = self.content.recommend(
                movie_title,
                top_n=top_n * 3,
                decade_filter=decade_filter,
                seen_movies=self.seen_movies
            )
            if not content_df.empty:
                # Normalise similarity 0-1
                mx = content_df["similarity"].max()
                content_df["content_score"] = (
                    content_df["similarity"] / mx if mx > 0 else 0
                )

        if not use_hybrid or self.collab is None:
            df = content_df.head(top_n).copy()
            df = self._apply_mood_filter(df, mood_genres)
            df = self._add_score_column(df)
            return df.head(top_n)

        # ── KNN Collab ───────────────────────────────────────────────
        knn_df = self.collab.recommend_knn(
            movie_title,
            top_n=top_n * 3,
            seen_movies=self.seen_movies
        )
        if not knn_df.empty:
            mx = knn_df["similarity"].max()
            knn_df["knn_score"] = knn_df["similarity"] / mx if mx > 0 else 0

        # ── SVD Collab ───────────────────────────────────────────────
        svd_df = pd.DataFrame()
        if user_id is not None:
            svd_df = self.collab.recommend_svd(
                user_id,
                top_n=top_n * 3,
                seen_movies=self.seen_movies
            )
            if not svd_df.empty:
                mx = svd_df["predicted_rating"].max()
                svd_df["svd_score"] = (
                    svd_df["predicted_rating"] / mx if mx > 0 else 0
                )

        # ── Merge & Blend ────────────────────────────────────────────
        blended = self._blend(content_df, knn_df, svd_df)
        blended = self._apply_mood_filter(blended, mood_genres)

        if decade_filter:
            blended = blended[blended["year"] == decade_filter // 10 * 10 + \
                              (blended["year"] % 10)]
            # Simpler: just filter by decade
            blended = blended[
                (blended["year"] >= decade_filter) &
                (blended["year"] < decade_filter + 10)
            ]

        return blended.head(top_n)

    def _blend(
        self,
        content_df: pd.DataFrame,
        knn_df    : pd.DataFrame,
        svd_df    : pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all three result sets by title and compute weighted hybrid score.
        """
        # Collect all unique titles
        all_titles = set()
        for df in [content_df, knn_df, svd_df]:
            if not df.empty and "title" in df.columns:
                all_titles.update(df["title"].tolist())

        rows = []
        for title in all_titles:
            c_score = 0.0
            k_score = 0.0
            s_score = 0.0
            row_data = {}

            if not content_df.empty:
                r = content_df[content_df["title"] == title]
                if not r.empty:
                    c_score  = float(r["content_score"].values[0])
                    row_data = r.iloc[0].to_dict()

            if not knn_df.empty:
                r = knn_df[knn_df["title"] == title]
                if not r.empty:
                    k_score  = float(r["knn_score"].values[0])
                    if not row_data:
                        row_data = r.iloc[0].to_dict()

            if not svd_df.empty:
                r = svd_df[svd_df["title"] == title]
                if not r.empty:
                    s_score  = float(r["svd_score"].values[0])
                    if not row_data:
                        row_data = r.iloc[0].to_dict()

            hybrid_score = (
                self.w_content * c_score +
                self.w_knn     * k_score +
                self.w_svd     * s_score
            )

            row_data["hybrid_score"]  = round(hybrid_score, 4)
            row_data["content_score"] = round(c_score, 4)
            row_data["knn_score"]     = round(k_score, 4)
            row_data["svd_score"]     = round(s_score, 4)
            row_data["source"]        = "Hybrid"
            rows.append(row_data)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        return df

    def _apply_mood_filter(
        self,
        df        : pd.DataFrame,
        mood_genres: list
    ) -> pd.DataFrame:
        """Filter + boost movies that match the selected mood genres."""
        if not mood_genres or df.empty or "genres" not in df.columns:
            return df

        def mood_match(genres_str):
            genres = str(genres_str).split("|")
            return any(g in mood_genres for g in genres)

        mask = df["genres"].apply(mood_match)
        matched   = df[mask].copy()
        unmatched = df[~mask].copy()
        # Boost matched movies' score slightly
        if "hybrid_score" in matched.columns:
            matched["hybrid_score"] = (matched["hybrid_score"] * 1.15).clip(upper=1.0)
        return pd.concat([matched, unmatched], ignore_index=True)

    def _add_score_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a display score column exists."""
        if "hybrid_score" not in df.columns and "similarity" in df.columns:
            df["hybrid_score"] = df["similarity"]
        return df

    # ------------------------------------------------------------------
    # EXPLAIN — "Why This?"
    # ------------------------------------------------------------------

    def explain(
        self,
        query_title: str,
        rec_title  : str,
        user_id    : Optional[int] = None
    ) -> dict:
        """
        Full explanation combining content + KNN signals.
        Powers the 'Why This?' feature in the Streamlit app.
        """
        explanation = {"movie": rec_title, "reasons": []}

        # Content explanation
        if self.content is not None:
            c_exp = self.content.explain(query_title, rec_title)
            if c_exp:
                shared = c_exp.get("shared_genres", [])
                overlap = c_exp.get("genre_overlap", 0)
                sim     = c_exp.get("similarity_score", 0)
                if shared:
                    explanation["reasons"].append(
                        f"✅ Shares genres: {', '.join(shared)} "
                        f"({overlap}% genre overlap)"
                    )
                explanation["content_similarity"] = sim
                explanation["shared_genres"]      = shared
                explanation["genre_overlap"]      = overlap

        # KNN explanation
        if self.collab is not None:
            k_exp = self.collab.explain_knn(query_title)
            neighbors = k_exp.get("neighbors", [])
            if neighbors:
                top_neighbor = neighbors[0]
                explanation["reasons"].append(
                    f"👥 Users who loved '{query_title}' gave this movie "
                    f"similar high ratings (KNN similarity: "
                    f"{top_neighbor['similarity']})"
                )
            explanation["knn_neighbors"] = neighbors

        # Rating quality
        if self.content is not None and self.content.movies is not None:
            rec_row = self.content.movies[
                self.content.movies["title"] == rec_title
            ]
            if not rec_row.empty:
                avg  = rec_row["avg_rating"].values[0]
                cnt  = rec_row["rating_count"].values[0]
                explanation["avg_rating"]   = round(float(avg), 2)
                explanation["rating_count"] = int(cnt)
                if avg >= 4.0:
                    explanation["reasons"].append(
                        f"⭐ Highly rated: {avg:.1f}/5 from {cnt:,} users"
                    )

        if not explanation["reasons"]:
            explanation["reasons"].append(
                "🎬 Similar content profile and viewing patterns"
            )

        return explanation

    # ------------------------------------------------------------------
    # SEEN-IT / TASTE PROFILE
    # ------------------------------------------------------------------

    def mark_seen(self, title: str):
        """Mark a movie as watched — exclude from future recs."""
        if title not in self.seen_movies:
            self.seen_movies.append(title)

    def unmark_seen(self, title: str):
        """Remove from watched list."""
        if title in self.seen_movies:
            self.seen_movies.remove(title)

    def rate_movie(self, title: str, rating: float):
        """Record a user rating for taste profile building."""
        self.taste_profile[title] = rating

    def get_mood_options(self) -> list:
        return list(MOOD_GENRE_MAP.keys())

    def get_decade_options(self, movies_df: pd.DataFrame) -> list:
        decades = sorted(movies_df["decade"].dropna().unique().astype(int).tolist())
        return ["All Eras"] + [f"{d}s" for d in decades]
