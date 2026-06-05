"""
data_loader.py
--------------
Loads and preprocesses MovieLens datasets.

Supports BOTH formats:
  - MovieLens 1M  (.dat files, '::' separator)  — legacy
  - MovieLens 25M / Latest (.csv files, ',' separator)  — current

Auto-detects which format is present in the data/ folder.
"""

import pandas as pd
import numpy as np
import os


def _detect_format(data_dir: str) -> str:
    """
    Auto-detect dataset format based on files present in data_dir.
    Returns 'csv' or 'dat'.
    """
    if os.path.exists(os.path.join(data_dir, "movies.csv")):
        return "csv"
    elif os.path.exists(os.path.join(data_dir, "movies.dat")):
        return "dat"
    else:
        raise FileNotFoundError(
            f"No dataset found in '{data_dir}'.\n"
            "Please place either:\n"
            "  • movies.csv + ratings.csv   (MovieLens 25M / Latest)\n"
            "  • movies.dat + ratings.dat   (MovieLens 1M)\n"
            f"Download from: https://grouplens.org/datasets/movielens/"
        )


def load_movies(data_dir: str = "data") -> pd.DataFrame:
    """
    Load movies from whichever format is present.
    Output always has columns: movieId, title, genres, genres_clean, year, decade
    """
    fmt = _detect_format(data_dir)

    if fmt == "csv":
        path = os.path.join(data_dir, "movies.csv")
        movies = pd.read_csv(path)
        # MovieLens CSV already has header: movieId,title,genres
    else:
        path = os.path.join(data_dir, "movies.dat")
        movies = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1"
        )

    # genres_clean: pipe-separated → space-separated (for TF-IDF vectorizer)
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)

    # Extract year from title e.g. "Toy Story (1995)" → 1995
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

    # Decade column for era filtering
    movies["decade"] = (movies["year"] // 10 * 10).astype("Int64")

    return movies


def load_ratings(data_dir: str = "data") -> pd.DataFrame:
    """
    Load ratings from whichever format is present.
    Output always has columns: userId, movieId, rating, timestamp
    """
    fmt = _detect_format(data_dir)

    if fmt == "csv":
        path = os.path.join(data_dir, "ratings.csv")
        print("   (this may take a moment for large rating files...)")
        ratings = pd.read_csv(path)
        # MovieLens CSV already has header: userId,movieId,rating,timestamp

        # For very large datasets (25M / 32M) sample down to keep training fast
        # while still covering all movies well.
        # 5 million ratings is plenty for SVD/KNN quality.
        MAX_RATINGS = 5_000_000
        if len(ratings) > MAX_RATINGS:
            print(f"   ℹ️  Large dataset ({len(ratings):,} ratings) — "
                  f"sampling {MAX_RATINGS:,} for faster training...")
            ratings = ratings.sample(n=MAX_RATINGS, random_state=42).reset_index(drop=True)

    else:
        path = os.path.join(data_dir, "ratings.dat")
        ratings = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1"
        )

    return ratings


def load_users(data_dir: str = "data") -> pd.DataFrame:
    """
    Load users — only available in MovieLens 1M (.dat format).
    Returns empty DataFrame for CSV-format datasets (25M / Latest).
    """
    fmt = _detect_format(data_dir)

    if fmt == "dat":
        path = os.path.join(data_dir, "users.dat")
        users = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zip"],
            encoding="latin-1"
        )
        return users
    else:
        # Newer datasets don't ship a users file — return empty frame
        return pd.DataFrame(columns=["userId"])


def compute_movie_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Merge average rating and rating count into the movies DataFrame.
    """
    stats = ratings.groupby("movieId").agg(
        avg_rating   = ("rating", "mean"),
        rating_count = ("rating", "count")
    ).reset_index()

    movies = movies.merge(stats, on="movieId", how="left")
    movies["avg_rating"]   = movies["avg_rating"].round(2).fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    return movies


def load_all(data_dir: str = "data"):
    """
    Master loader — returns (movies, ratings, users) all preprocessed.
    Auto-detects dataset format.
    """
    fmt = _detect_format(data_dir)
    label = "MovieLens 25M/Latest (CSV)" if fmt == "csv" else "MovieLens 1M (DAT)"
    print(f"📂 Loading {label} dataset from '{data_dir}'...")

    movies  = load_movies(data_dir)
    ratings = load_ratings(data_dir)
    users   = load_users(data_dir)
    movies  = compute_movie_stats(movies, ratings)

    # Filter out movies with no ratings at all (they won't help the models)
    movies = movies[movies["rating_count"] > 0].reset_index(drop=True)

    print(f"✅ Movies  : {len(movies):,}")
    print(f"✅ Ratings : {len(ratings):,}")
    if not users.empty:
        print(f"✅ Users   : {len(users):,}")

    return movies, ratings, users
