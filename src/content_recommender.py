"""
content_recommender.py
-----------------------
Content-Based Filtering using:
  1. TF-IDF Vectorizer on genres
  2. Sentence Transformers (BERT) for semantic embeddings
  3. Cosine Similarity to rank movies

Why two methods?
  - TF-IDF is fast and interpretable (genre keyword matching)
  - BERT embeddings capture semantic meaning beyond exact keywords
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import joblib
import os


class ContentRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix     = None
        self.bert_matrix      = None   # filled if sentence-transformers available
        self.cosine_sim       = None   # final similarity matrix
        self.movies           = None
        self.movie_indices    = None   # title -> index lookup

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def fit(self, movies: pd.DataFrame, use_bert: bool = False):
        """
        Build the similarity matrix from movie genres (+ optional BERT).

        Parameters
        ----------
        movies    : DataFrame with columns [title, genres_clean, ...]
        use_bert  : if True, also build BERT embeddings and blend them
        """
        self.movies = movies.reset_index(drop=True)
        self.movie_indices = pd.Series(
            self.movies.index, index=self.movies["title"]
        )

        print("🔤 Building TF-IDF matrix on genres...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies["genres_clean"].fillna("")
        )
        tfidf_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        if use_bert:
            bert_sim = self._build_bert_similarity()
            # Weighted blend: 40% TF-IDF genre, 60% semantic BERT
            self.cosine_sim = 0.4 * tfidf_sim + 0.6 * bert_sim
            print("✅ Content model ready (TF-IDF + BERT blend)")
        else:
            self.cosine_sim = tfidf_sim
            print("✅ Content model ready (TF-IDF only)")

        return self

    def _build_bert_similarity(self):
        """Build semantic similarity using Sentence Transformers (BERT)."""
        try:
            from sentence_transformers import SentenceTransformer
            print("🧠 Loading Sentence Transformer (this may take a moment)...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Encode genres as natural language sentences
            genre_sentences = self.movies["genres"].str.replace("|", " ", regex=False)
            embeddings = model.encode(
                genre_sentences.tolist(),
                show_progress_bar=True,
                batch_size=64
            )
            embeddings = normalize(embeddings)
            sim = np.dot(embeddings, embeddings.T)
            self.bert_matrix = embeddings
            print("✅ BERT embeddings built")
            return sim
        except ImportError:
            print("⚠️  sentence-transformers not installed — using TF-IDF only")
            tfidf_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            return tfidf_sim

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------

    def recommend(
        self,
        movie_title: str,
        top_n: int = 10,
        decade_filter: int = None,
        seen_movies: list = None
    ) -> pd.DataFrame:
        """
        Return top_n content-based recommendations.

        Parameters
        ----------
        movie_title   : exact movie title string
        top_n         : number of results
        decade_filter : e.g. 1990 → only movies from 1990s
        seen_movies   : list of titles to exclude (already watched)
        """
        if movie_title not in self.movie_indices:
            return pd.DataFrame()   # caller handles not-found

        idx  = self.movie_indices[movie_title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]   # exclude self

        results = []
        for i, score in sim_scores:
            row = self.movies.iloc[i]
            # Era filter
            if decade_filter and row.get("decade") != decade_filter:
                continue
            # Seen filter
            if seen_movies and row["title"] in seen_movies:
                continue
            results.append({
                "title"        : row["title"],
                "genres"       : row["genres"],
                "year"         : row.get("year", "N/A"),
                "avg_rating"   : row.get("avg_rating", 0),
                "rating_count" : row.get("rating_count", 0),
                "similarity"   : round(score, 4),
                "source"       : "Content (TF-IDF/BERT)"
            })
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def explain(self, movie_title: str, recommended_title: str) -> dict:
        """
        Explain WHY a movie was recommended.
        Returns genre overlap details.
        """
        if movie_title not in self.movie_indices or \
           recommended_title not in self.movie_indices:
            return {}

        idx1 = self.movie_indices[movie_title]
        idx2 = self.movie_indices[recommended_title]

        g1 = set(self.movies.iloc[idx1]["genres"].split("|"))
        g2 = set(self.movies.iloc[idx2]["genres"].split("|"))

        shared   = g1 & g2
        only_g1  = g1 - g2
        only_g2  = g2 - g1
        overlap  = round(len(shared) / len(g1 | g2) * 100, 1) if g1 | g2 else 0
        sim      = round(float(self.cosine_sim[idx1][idx2]), 4)

        return {
            "shared_genres"  : list(shared),
            "unique_to_query": list(only_g1),
            "unique_to_rec"  : list(only_g2),
            "genre_overlap"  : overlap,
            "similarity_score": sim
        }

    # ------------------------------------------------------------------
    # PERSIST
    # ------------------------------------------------------------------

    def save(self, path: str = "models/content_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 Content model saved → {path}")

    @staticmethod
    def load(path: str = "models/content_model.pkl"):
        model = joblib.load(path)
        print(f"📦 Content model loaded ← {path}")
        return model
