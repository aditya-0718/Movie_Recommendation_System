# 🎬 CineMatch AI — Movie Recommendation System

A hybrid AI-powered movie recommendation system built with Python and Streamlit. Combines Content-Based Filtering (TF-IDF/BERT), Collaborative Filtering (SVD + KNN), and a mood-aware engine for personalized recommendations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

- **Hybrid Recommendation Engine** — blends three algorithms for best accuracy:
  - 📝 Content-Based (TF-IDF genre similarity)
  - 🔢 SVD Matrix Factorization (latent factor discovery)
  - 🔗 KNN Item-Based Collaborative Filtering (sparse, memory-efficient for 25M+ ratings)
- **Mood-Aware Picks** — recommends movies based on your current mood (Adventurous, Sad, Stressed, etc.)
- **Explainable AI** — "Why This?" feature shows exactly why each movie was recommended
- **Taste Profile Builder** — rate movies to personalize your recommendations
- **Era Filtering** — filter by decade (1970s–2000s)
- **Watched List** — mark movies as seen to avoid duplicates
- **EDA Dashboard** — interactive charts: genre distribution, rating histogram, movies per decade

---

## 🗂️ Project Structure

```
Movie_Recommendation_System/
│
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
│
├── src/
│   ├── data_loader.py          # Dataset loading & preprocessing
│   ├── content_recommender.py  # TF-IDF / BERT content-based model
│   ├── collab_recommender.py   # SVD + KNN collaborative model
│   └── hybrid_recommender.py  # Hybrid blending engine
│
├── data/                   # MovieLens dataset files (not committed)
│   ├── movies.dat / movies.csv
│   ├── ratings.dat / ratings.csv
│   └── users.dat (1M only)
│
└── models/                 # Trained model files (not committed)
    ├── content_model.pkl
    └── collab_model.pkl
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Movie_Recommendation_System.git
cd Movie_Recommendation_System
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download **MovieLens 1M** or **MovieLens 25M** from [GroupLens](https://grouplens.org/datasets/movielens/) and place the files in the `data/` folder:

| Format | Files needed |
|--------|-------------|
| MovieLens 1M | `movies.dat`, `ratings.dat`, `users.dat` |
| MovieLens 25M | `movies.csv`, `ratings.csv` |

The data loader auto-detects which format is present.

### 5. Train the models
```bash
python train.py
```

Optional flags:
```bash
python train.py --bert        # Use BERT embeddings (slower, more accurate)
python train.py --test        # Run a quick sanity check after training
python train.py --data data/  # Specify a custom data directory
```

### 6. Launch the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

### Hybrid Scoring

Each movie receives a weighted hybrid score:

```
Hybrid Score = 0.30 × Content Score + 0.40 × SVD Score + 0.30 × KNN Score
```

| Model | Weight | Method |
|-------|--------|--------|
| Content (TF-IDF/BERT) | 30% | Genre cosine similarity |
| SVD | 40% | Matrix factorization of user-item ratings |
| KNN | 30% | Item-based nearest neighbor (sparse cosine, sklearn) |

> **Large dataset note:** KNN uses a sparse `csr_matrix` via scikit-learn instead of Surprise's dense similarity matrix. This avoids the ~11 GiB memory allocation that occurs with 25M-scale datasets (39k× 39k item matrix). Only items with ≥ 50 ratings are indexed, keeping memory well under 1 GiB.

### Mood → Genre Mapping

| Mood | Matched Genres |
|------|---------------|
| 😢 Sad / Need Comfort | Comedy, Animation, Family, Musical |
| 🤩 Adventurous | Action, Adventure, Sci-Fi, Fantasy |
| 😱 Want Thrills | Thriller, Horror, Mystery, Crime |
| 🧠 Want to Think | Sci-Fi, Drama, Mystery, Documentary |
| 💕 Romantic Mood | Romance, Drama |
| 🎉 Party / Fun | Comedy, Animation, Musical, Action |

---

## 📊 Dataset

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) by GroupLens Research.

- **1M version**: ~1M ratings, 6K movies, 4K users
- **25M version**: ~25M ratings, 62K movies, 162K users (recommended)

> ⚠️ Dataset files are **not included** in this repository due to size. Download them separately from GroupLens.

---

## 🖥️ App Screenshots

| Tab | Description |
|-----|-------------|
| 🔍 Recommend | Search a movie, get hybrid recommendations with score breakdown |
| 🎭 Mood Picks | Pick your mood, get curated suggestions |
| 📊 Dashboard | EDA charts — genres, ratings, decades |
| 🎯 Taste Profile | Rate movies to personalize results |

---

## 🔧 Configuration

You can adjust the hybrid weights in `src/hybrid_recommender.py`:

```python
HybridRecommender(
    w_content = 0.30,   # Content-based weight
    w_svd     = 0.40,   # SVD weight
    w_knn     = 0.30    # KNN weight
)
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.0 | Web UI |
| pandas | 2.1.0 | Data manipulation |
| scikit-learn | 1.3.0 | TF-IDF, cosine similarity |
| scikit-surprise | 1.1.3 | SVD collaborative filtering |
| plotly | 5.17.0 | Interactive charts |
| sentence-transformers | 2.2.2 | BERT embeddings (optional) |
| joblib | 1.3.2 | Model serialization |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset
- [Surprise library](http://surpriselib.com/) for collaborative filtering algorithms
- [Streamlit](https://streamlit.io/) for the web framework
