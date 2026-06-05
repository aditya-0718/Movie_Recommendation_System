"""
train.py
--------
Train all models (Content, SVD, KNN) and save to disk.

Usage:
    python train.py                  # standard training
    python train.py --bert           # also build BERT embeddings (slower)
    python train.py --data data/     # custom data directory
    python train.py --test           # run quick test after training
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader         import load_all
from content_recommender import ContentRecommender
from collab_recommender  import CollabRecommender
from hybrid_recommender  import HybridRecommender


def train(data_dir: str = "data", use_bert: bool = False):
    t0 = time.time()

    # ── 1. Load Data ────────────────────────────────────────────────
    movies, ratings, users = load_all(data_dir)

    # ── 2. Content Model ────────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1/2 — Content-Based Model (TF-IDF / BERT)")
    print("="*55)
    content_model = ContentRecommender()
    content_model.fit(movies, use_bert=use_bert)
    content_model.save("models/content_model.pkl")

    # ── 3. Collaborative Models (SVD + KNN) ─────────────────────────
    print("\n" + "="*55)
    print("STEP 2/2 — Collaborative Models (SVD + KNN)")
    print("  Note: with 25M dataset this step takes ~5-10 minutes")
    print("="*55)
    collab_model = CollabRecommender(k_neighbors=20)
    collab_model.fit(movies, ratings)
    collab_model.save("models/collab_model.pkl")

    elapsed = round(time.time() - t0, 1)
    print(f"\n🎉 All models trained and saved in {elapsed}s")
    print("👉 Now run:  streamlit run app.py")


def quick_test(data_dir: str = "data"):
    """Sanity check — load models and print sample recommendations."""
    print("\n🧪 Running quick test...")

    content = ContentRecommender.load("models/content_model.pkl")
    collab  = CollabRecommender.load("models/collab_model.pkl")
    hybrid  = HybridRecommender(content_model=content, collab_model=collab)

    # Pick test movies that exist in both 1M and 25M
    test_movies = ["Toy Story (1995)", "Titanic (1997)", "Matrix, The (1999)",
                   "Inception (2010)", "Interstellar (2014)"]

    for movie in test_movies:
        recs = hybrid.recommend(movie, top_n=5)
        if recs.empty:
            print(f"  ⚠️  '{movie}' not found — skipping")
            continue
        print(f"\n🎬 Recommendations for: {movie}")
        print("-" * 50)
        for i, row in recs.iterrows():
            score = row.get("hybrid_score", row.get("similarity", 0))
            print(f"  {i+1}. {row['title']}  |  Score: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Movie Recommender Models")
    parser.add_argument("--data", default="data", help="Path to data directory")
    parser.add_argument("--bert", action="store_true",
                        help="Use BERT embeddings (requires sentence-transformers)")
    parser.add_argument("--test", action="store_true",
                        help="Run quick sanity test after training")
    args = parser.parse_args()

    train(data_dir=args.data, use_bert=args.bert)

    if args.test:
        quick_test(args.data)
