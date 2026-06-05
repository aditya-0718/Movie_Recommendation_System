"""
app.py  —  CineMatch AI
========================
All bugs properly fixed:

FIX A — "Why This?" / "Mark Seen" buttons:
  Root cause: recs DataFrame lived inside `if recommend_btn:` block.
  On any button click Streamlit reruns; recommend_btn becomes False so the
  whole result block (including button logic) never executes again — buttons
  render but their callbacks are dead.
  Fix: store recs in st.session_state["last_recs"] and
       st.session_state["last_query"] so results persist across reruns.

FIX B — Taste Profile N-1 (off-by-one):
  Root cause: `st.session_state["taste_profile"] = updated_profile` wiped
  the profile on every rerun. During the rerun triggered by any button click
  the slider widget values briefly read as their default (0) before Streamlit
  restores widget state, so the overwrite zeroed everything.
  Fix: never overwrite the whole dict. Instead merge per-slider:
    - if slider > 0  → write that key into session_state["taste_profile"]
    - if slider == 0 → delete that key from session_state["taste_profile"]
  This way each slider independently manages its own key without touching
  the others, so clicking any button anywhere on the page cannot wipe ratings.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .main-title {
    font-size: 3rem; font-weight: 900; text-align: center;
    background: linear-gradient(90deg, #f5c518, #e50914, #f5c518);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { text-align:center; color:#8b949e; font-size:1.1rem; margin-bottom:2rem; }
  .movie-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:1.2rem; margin:0.5rem 0;
  }
  .movie-card:hover { border-color:#f5c518; }
  .movie-title { font-size:1.1rem; font-weight:700; color:#f5c518; }
  .movie-meta  { font-size:0.85rem; color:#8b949e; margin-top:0.3rem; }
  .score-badge {
    display:inline-block; background:#21262d; border:1px solid #30363d;
    border-radius:20px; padding:0.2rem 0.7rem; font-size:0.8rem;
    color:#3fb950; font-weight:600;
  }
  .genre-tag {
    display:inline-block; background:#1f3a5f; border-radius:4px;
    padding:0.1rem 0.5rem; font-size:0.75rem; color:#58a6ff; margin:0.1rem;
  }
  .seen-badge {
    display:inline-block; background:#3d1f1f; border-radius:4px;
    padding:0.1rem 0.5rem; font-size:0.75rem; color:#f85149;
  }
  .why-box {
    background:#0d2818; border:1px solid #238636;
    border-radius:8px; padding:1rem; margin-top:0.5rem; font-size:0.88rem;
  }
  .stButton>button {
    background:#e50914; color:white; border:none;
    border-radius:8px; font-weight:700;
  }
  .stButton>button:hover { background:#c40812; }
  .section-header {
    font-size:1.3rem; font-weight:700; color:#f5c518;
    border-bottom:2px solid #21262d; padding-bottom:0.4rem;
    margin:1.5rem 0 1rem 0;
  }
  div[data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:1rem;
  }
  .algo-box { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:0.9rem; margin-bottom:0.5rem; }
  .algo-box h4 { color:#f5c518; margin:0 0 0.3rem 0; }
  .algo-box p  { color:#8b949e; font-size:0.82rem; margin:0; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "seen_movies"       : [],
        "taste_profile"     : {},
        "hybrid"            : None,
        "movies_df"         : None,
        "models_loaded"     : False,
        "why_open"          : {},     # card_id → bool
        "taste_movies"      : [],
        "taste_initialized" : False,
        "mood_tab_mood"     : None,
        # FIX A — persist recommendation results across reruns
        "last_recs"         : None,   # pd.DataFrame or None
        "last_query"        : None,   # str or None
        "last_use_hybrid"   : True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from content_recommender import ContentRecommender
    from collab_recommender  import CollabRecommender
    from hybrid_recommender  import HybridRecommender
    content = ContentRecommender.load("models/content_model.pkl")
    collab  = CollabRecommender.load("models/collab_model.pkl")
    hybrid  = HybridRecommender(content_model=content, collab_model=collab)
    return hybrid, content.movies


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 CineMatch AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hybrid AI Recommendations · Mood-Aware · Explainable</div>',
    unsafe_allow_html=True
)

# ── Models check ──────────────────────────────────────────────────────────────
models_exist = (
    os.path.exists("models/content_model.pkl") and
    os.path.exists("models/collab_model.pkl")
)
if not models_exist:
    st.error("⚠️ Models not found! Please run: `python train.py`", icon="🚨")
    st.code("python train.py", language="bash")
    st.stop()

with st.spinner("🤖 Loading AI models..."):
    hybrid, movies_df = load_models()
    st.session_state["hybrid"]        = hybrid
    st.session_state["movies_df"]     = movies_df
    st.session_state["models_loaded"] = True
    hybrid.seen_movies   = st.session_state["seen_movies"]
    hybrid.taste_profile = st.session_state["taste_profile"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("### 🤖 Algorithm")
    algo = st.radio(
        "Choose recommendation engine:",
        ["🔀 Hybrid (Best)", "📝 Content Only"],
    )
    use_hybrid = algo == "🔀 Hybrid (Best)"

    if use_hybrid:
        st.markdown("""<div class="algo-box"><h4>🔀 Hybrid</h4>
        <p>Blends genre matching (30%) + SVD rating patterns (40%)
        + KNN user taste (30%). Best accuracy. Recommended.</p></div>""",
        unsafe_allow_html=True)
    else:
        st.markdown("""<div class="algo-box"><h4>📝 Content Only</h4>
        <p>TF-IDF genre similarity only. Fast and transparent.</p></div>""",
        unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎭 Your Mood")
    mood = st.selectbox("How are you feeling?", hybrid.get_mood_options())

    st.divider()
    st.markdown("### 📅 Era Filter")
    era = st.selectbox("Filter by decade", ["All Eras","1970s","1980s","1990s","2000s"])
    decade_filter = None if era == "All Eras" else int(era[:4])

    st.divider()
    st.markdown("### 🚫 Watched List")
    if st.session_state["seen_movies"]:
        for m in st.session_state["seen_movies"][:5]:
            c1, c2 = st.columns([4,1])
            c1.markdown(f"<small>🎬 {m[:25]}...</small>", unsafe_allow_html=True)
            if c2.button("✕", key=f"rm_{m}"):
                st.session_state["seen_movies"].remove(m)
                st.rerun()
        if len(st.session_state["seen_movies"]) > 5:
            st.caption(f"...and {len(st.session_state['seen_movies'])-5} more")
    else:
        st.caption("No movies marked as watched yet.")

    st.divider()
    st.markdown("### 🎯 Taste Profile")
    if st.session_state["taste_profile"]:
        for t, r in list(st.session_state["taste_profile"].items())[:3]:
            st.caption(f"⭐ {r}/5 — {t[:20]}")
        extra = len(st.session_state["taste_profile"]) - 3
        if extra > 0:
            st.caption(f"...and {extra} more")
    else:
        st.caption("Rate movies in the Taste Profile tab.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Recommend", "🎭 Mood Picks", "📊 Dashboard", "🎯 Taste Profile"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS   (FIX A applied here)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    all_titles = sorted(movies_df["title"].tolist())

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        selected_movie = st.selectbox(
            "🎬 Search for a movie",
            options=[None] + all_titles,
            index=0,
            format_func=lambda x: "🎬 Type a movie name..." if x is None else x,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        recommend_btn = st.button("🎯 Recommend", use_container_width=True)

    top_n = st.slider("Number of recommendations", 5, 20, 10)

    # ── FIX A: Run recommendation only when button clicked, then STORE in session_state
    if recommend_btn:
        if selected_movie is None:
            st.warning("⚠️ Please select a movie first!")
            st.session_state["last_recs"]  = None
            st.session_state["last_query"] = None
        else:
            hybrid.seen_movies = st.session_state["seen_movies"]
            with st.spinner(f"🤖 Finding movies similar to **{selected_movie}**..."):
                recs = hybrid.recommend(
                    movie_title   = selected_movie,
                    top_n         = top_n,
                    mood          = mood,
                    decade_filter = decade_filter,
                    use_hybrid    = use_hybrid
                )
            # Store everything needed to render results on subsequent reruns
            st.session_state["last_recs"]        = recs
            st.session_state["last_query"]       = selected_movie
            st.session_state["last_use_hybrid"]  = use_hybrid
            # Reset why_open for new result set
            st.session_state["why_open"] = {}

    # ── Always render results from session_state (survives button reruns) ────
    recs       = st.session_state.get("last_recs")
    last_query = st.session_state.get("last_query")
    last_hybrid= st.session_state.get("last_use_hybrid", True)

    if recs is not None and not recs.empty and last_query:

        # Query movie header card
        query_row = movies_df[movies_df["title"] == last_query]
        if not query_row.empty:
            qr = query_row.iloc[0]
            genre_tags_q = "".join(
                f'<span class="genre-tag">{g}</span>'
                for g in qr["genres"].split("|")
            )
            st.markdown(f"""
            <div class="movie-card" style="border-color:#f5c518;margin-bottom:1.5rem;">
              <div style="font-size:0.8rem;color:#8b949e;">You searched for:</div>
              <div class="movie-title" style="font-size:1.4rem;">{qr['title']}</div>
              <div class="movie-meta">{genre_tags_q}</div>
              <div class="movie-meta" style="margin-top:0.5rem;">
                ⭐ {qr['avg_rating']:.1f}/5 &nbsp;|&nbsp;
                👥 {int(qr['rating_count']):,} ratings &nbsp;|&nbsp;
                📅 {int(qr['year']) if not pd.isna(qr['year']) else 'N/A'}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="section-header">🎯 Top {len(recs)} Recommendations</div>',
            unsafe_allow_html=True
        )

        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            title   = row["title"]
            genres  = row.get("genres", "")
            year    = row.get("year", "")
            rating  = row.get("avg_rating", 0)
            count   = row.get("rating_count", 0)
            score   = row.get("hybrid_score", row.get("similarity", 0))
            is_seen = title in st.session_state["seen_movies"]

            # Stable key: title only (not rank — rank can change if seen filter applied)
            card_id = f"rec__{title}"
            if card_id not in st.session_state["why_open"]:
                st.session_state["why_open"][card_id] = False

            genre_tags = "".join(
                f'<span class="genre-tag">{g}</span>'
                for g in str(genres).split("|")
            )
            seen_badge = '<span class="seen-badge">👁 Watched</span>' if is_seen else ""

            st.markdown(f"""
            <div class="movie-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div class="movie-title">#{rank} {title} {seen_badge}</div>
                  <div class="movie-meta">{genre_tags}</div>
                  <div class="movie-meta" style="margin-top:0.4rem;">
                    ⭐ {float(rating):.1f}/5 &nbsp;|&nbsp;
                    👥 {int(count):,} ratings &nbsp;|&nbsp;
                    📅 {int(year) if year and not pd.isna(year) else 'N/A'}
                  </div>
                </div>
                <div><span class="score-badge">Score: {float(score):.3f}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2, _ = st.columns([2, 2, 4])

            # ── MARK SEEN (FIX A) ────────────────────────────────────────────
            # Button click → mutate session_state → rerun.
            # On rerun recommend_btn=False but last_recs is still in session_state
            # so results re-render correctly with updated seen status.
            with btn_col1:
                label_seen = "↩ Unmark Seen" if is_seen else "👁 Mark Seen"
                if st.button(label_seen, key=f"seen__{card_id}"):
                    if is_seen:
                        st.session_state["seen_movies"].remove(title)
                    else:
                        if title not in st.session_state["seen_movies"]:
                            st.session_state["seen_movies"].append(title)
                    st.rerun()

            # ── WHY THIS? (FIX A) ────────────────────────────────────────────
            # Toggle why_open[card_id] → rerun → result block re-renders
            # → why_open[card_id] is True → explanation box appears.
            with btn_col2:
                why_label = "✖ Hide Why" if st.session_state["why_open"][card_id] else "💡 Why This?"
                if st.button(why_label, key=f"why__{card_id}"):
                    st.session_state["why_open"][card_id] = not st.session_state["why_open"][card_id]
                    st.rerun()

            if st.session_state["why_open"][card_id]:
                with st.spinner("Generating explanation..."):
                    exp = hybrid.explain(last_query, title)

                why_html = "".join(f"<div>• {r}</div>" for r in exp.get("reasons", []))

                score_detail = ""
                if last_hybrid:
                    cs = row.get("content_score", 0)
                    ks = row.get("knn_score", 0)
                    ss = row.get("svd_score", 0)
                    score_detail = (
                        f'<div style="margin-top:0.5rem;font-size:0.8rem;color:#8b949e;">'
                        f'📐 Content: {float(cs):.3f} &nbsp;|&nbsp;'
                        f'🔗 KNN: {float(ks):.3f} &nbsp;|&nbsp;'
                        f'🔢 SVD: {float(ss):.3f}</div>'
                    )

                st.markdown(f"""
                <div class="why-box">
                  <strong>💡 Why "{title}" was recommended:</strong>
                  <div style="margin-top:0.5rem;">{why_html}</div>
                  {score_detail}
                </div>
                """, unsafe_allow_html=True)

    elif recs is not None and recs.empty:
        st.warning("No recommendations found. Try removing filters or choosing a different movie.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MOOD PICKS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎭 Pick Your Mood</div>', unsafe_allow_html=True)
    st.markdown("No specific movie in mind? Select how you're feeling and we'll pick for you.")

    MOOD_DEFS = [
        {"emoji":"😢","label":"Sad / Need Comfort",    "key":"😢 Sad / Need Comfort",     "desc":"Feel-good uplifting movies",    "genres":["Comedy","Animation","Family","Musical"]},
        {"emoji":"😤","label":"Stressed",               "key":"😤 Stressed / Overwhelmed",  "desc":"Easy and fun unwind",           "genres":["Comedy","Animation","Romance","Musical"]},
        {"emoji":"🤩","label":"Adventurous",            "key":"🤩 Adventurous",             "desc":"Epic action and big worlds",    "genres":["Action","Adventure","Sci-Fi","Fantasy"]},
        {"emoji":"😴","label":"Lazy Sunday",            "key":"😴 Lazy Sunday",             "desc":"Slow, warm, comfortable",       "genres":["Drama","Romance","Comedy"]},
        {"emoji":"😱","label":"Want Thrills",           "key":"😱 Want Thrills",            "desc":"Edge-of-seat suspense",         "genres":["Thriller","Horror","Mystery","Crime"]},
        {"emoji":"🧠","label":"Want to Think",          "key":"🧠 Want to Think",           "desc":"Mind-bending, thought-provoking","genres":["Sci-Fi","Drama","Mystery","Documentary"]},
        {"emoji":"💕","label":"Romantic Mood",          "key":"💕 Romantic Mood",           "desc":"Love stories and emotions",     "genres":["Romance","Drama"]},
        {"emoji":"🎉","label":"Party / Fun",            "key":"🎉 Party / Fun",             "desc":"High energy, laugh-out-loud",   "genres":["Comedy","Animation","Musical","Action"]},
    ]

    selected_mood_key = st.session_state.get("mood_tab_mood", None)
    cols = st.columns(4)
    for i, md in enumerate(MOOD_DEFS):
        with cols[i % 4]:
            is_active    = selected_mood_key == md["key"]
            border_color = "#f5c518" if is_active else "#30363d"
            bg_color     = "#1c2a1c" if is_active else "#161b22"
            st.markdown(f"""
            <div style="background:{bg_color};border:2px solid {border_color};
                        border-radius:12px;padding:1rem;text-align:center;margin-bottom:0.5rem;">
              <div style="font-size:2rem;">{md['emoji']}</div>
              <div style="font-weight:700;color:#e6edf3;font-size:0.9rem;">
                {"✅ " if is_active else ""}{md['label']}
              </div>
              <div style="font-size:0.75rem;color:#8b949e;margin-top:0.3rem;">{md['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(
                "✅ Selected" if is_active else "Select",
                key=f"mood_pick_{i}",
                use_container_width=True
            ):
                st.session_state["mood_tab_mood"] = None if is_active else md["key"]
                st.rerun()

    st.divider()

    if st.session_state["mood_tab_mood"]:
        active_def = next((m for m in MOOD_DEFS if m["key"] == st.session_state["mood_tab_mood"]), None)
        if active_def:
            genre_list = active_def["genres"]
            st.markdown(
                f'<div class="section-header">{active_def["emoji"]} Best for "{active_def["label"]}"</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Matching genres: {' · '.join(genre_list)}")
            mood_top_n = st.slider("How many to show?", 5, 30, 12, key="mood_top_n_slider")

            def has_mood_genre(g_str):
                return any(g in genre_list for g in str(g_str).split("|"))

            mood_movies = movies_df[
                movies_df["genres"].apply(has_mood_genre) &
                ~movies_df["title"].isin(st.session_state["seen_movies"]) &
                (movies_df["rating_count"] >= 50)
            ].copy().sort_values("avg_rating", ascending=False)

            if decade_filter:
                mood_movies = mood_movies[
                    (mood_movies["year"] >= decade_filter) &
                    (mood_movies["year"] < decade_filter + 10)
                ]

            mood_movies = mood_movies.head(mood_top_n)

            if mood_movies.empty:
                st.warning("No movies found. Try removing the Era filter.")
            else:
                for rank, (_, row) in enumerate(mood_movies.iterrows(), 1):
                    title   = row["title"]
                    genres  = row.get("genres","")
                    year    = row.get("year","")
                    rating  = row.get("avg_rating", 0)
                    count   = row.get("rating_count", 0)
                    is_seen = title in st.session_state["seen_movies"]
                    genre_tags = "".join(
                        f'<span class="genre-tag">{g}</span>'
                        for g in str(genres).split("|")
                    )
                    st.markdown(f"""
                    <div class="movie-card">
                      <div class="movie-title">#{rank} {title}</div>
                      <div class="movie-meta">{genre_tags}</div>
                      <div class="movie-meta" style="margin-top:0.4rem;">
                        ⭐ {float(rating):.1f}/5 &nbsp;|&nbsp;
                        👥 {int(count):,} ratings &nbsp;|&nbsp;
                        📅 {int(year) if year and not pd.isna(year) else 'N/A'}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    mc1, _, __ = st.columns([2, 2, 4])
                    with mc1:
                        lbl = "↩ Unmark Seen" if is_seen else "👁 Mark Seen"
                        if st.button(lbl, key=f"mood_seen__mood_{rank}__{title}"):
                            if is_seen:
                                st.session_state["seen_movies"].remove(title)
                            else:
                                if title not in st.session_state["seen_movies"]:
                                    st.session_state["seen_movies"].append(title)
                            st.rerun()
    else:
        st.info("👆 Select a mood above to get movie picks without needing to search for a title.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎬 Total Movies",  f"{len(movies_df):,}")
    m2.metric("⭐ Avg Rating",    f"{movies_df['avg_rating'].mean():.2f}")
    m3.metric("📅 Year Range",    f"{int(movies_df['year'].min())}–{int(movies_df['year'].max())}")
    m4.metric("🎭 Total Ratings", f"{movies_df['rating_count'].sum():,}")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        gc = {}
        for genres in movies_df["genres"].dropna():
            for g in genres.split("|"):
                gc[g] = gc.get(g, 0) + 1
        genre_df = pd.DataFrame(list(gc.items()), columns=["Genre","Count"]) \
                     .sort_values("Count", ascending=True).tail(15)
        fig1 = px.bar(genre_df, x="Count", y="Genre", orientation="h",
                      title="🎭 Movies per Genre", color="Count",
                      color_continuous_scale="Reds", template="plotly_dark")
        fig1.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        rb = movies_df[movies_df["avg_rating"] > 0]["avg_rating"]
        fig2 = px.histogram(rb, nbins=30, title="⭐ Rating Distribution",
                            labels={"value":"Average Rating","count":"Movies"},
                            color_discrete_sequence=["#f5c518"], template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    dd = movies_df.groupby("decade").size().reset_index(name="count") \
           .dropna().sort_values("decade")
    dd["decade"] = dd["decade"].astype(int).astype(str) + "s"
    fig3 = px.bar(dd, x="decade", y="count", title="📅 Movies per Decade",
                  color="count", color_continuous_scale="Blues", template="plotly_dark")
    fig3.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">🏆 Top Rated Movies</div>', unsafe_allow_html=True)
    top_rated = movies_df[movies_df["rating_count"] >= 100].nlargest(10, "avg_rating")[
        ["title","genres","avg_rating","rating_count","year"]
    ]
    st.dataframe(
        top_rated.style.format({"avg_rating":"{:.2f}","rating_count":"{:,}","year":"{:.0f}"}),
        use_container_width=True, hide_index=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — TASTE PROFILE   (FIX B applied here)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🎯 Build Your Taste Profile</div>', unsafe_allow_html=True)
    st.markdown(
        "Rate movies you've seen. Use **0** for unseen. "
        "Click **🔄 Swap** to replace any movie with one of your choice."
    )

    all_titles_set = set(movies_df["title"].tolist())

    # Initialise 10 slots once
    if not st.session_state["taste_initialized"]:
        popular = movies_df[movies_df["rating_count"] > 200].sample(
            min(10, len(movies_df[movies_df["rating_count"] > 200])),
            random_state=42
        )
        st.session_state["taste_movies"]      = popular["title"].tolist()
        st.session_state["taste_initialized"] = True

    st.markdown("### Rate these movies &nbsp; *(1 = Hated · 5 = Loved · 0 = Haven't seen)*")
    st.caption("Click **🔄 Swap** on any row to pick a different movie.")

    # ── FIX B: per-slider merge, never overwrite the whole dict ─────────────
    for slot_idx, movie in enumerate(st.session_state["taste_movies"]):
        row_data = movies_df[movies_df["title"] == movie]
        if row_data.empty:
            continue
        genres = row_data.iloc[0]["genres"]

        col_title, col_rate, col_swap = st.columns([3, 2, 1])

        with col_title:
            st.markdown(f"**{movie}**")
            st.caption(genres.replace("|", " · "))

        with col_rate:
            # Read saved value for THIS movie (default 0)
            saved_val = st.session_state["taste_profile"].get(movie, 0)
            new_rating = st.slider(
                "Rating", 0, 5, int(saved_val),
                key=f"taste_slider__{movie}",
                label_visibility="collapsed"
            )
            # FIX B: update ONLY this movie's key — never touch others
            if new_rating > 0:
                st.session_state["taste_profile"][movie] = new_rating
                hybrid.rate_movie(movie, new_rating)
            else:
                # Remove from profile if slider dragged back to 0
                st.session_state["taste_profile"].pop(movie, None)

        with col_swap:
            if st.button("🔄 Swap", key=f"swap_open__{slot_idx}"):
                swap_key = f"swap_active__{slot_idx}"
                st.session_state[swap_key] = not st.session_state.get(swap_key, False)
                st.rerun()

        swap_key = f"swap_active__{slot_idx}"
        if st.session_state.get(swap_key, False):
            other_slots = set(st.session_state["taste_movies"]) - {movie}
            available   = [t for t in sorted(all_titles_set) if t not in other_slots]
            new_movie = st.selectbox(
                f"Replace **{movie}** with:",
                options=[None] + available,
                format_func=lambda x: "— choose a movie —" if x is None else x,
                key=f"swap_select__{slot_idx}"
            )
            cc1, cc2, _ = st.columns([1, 1, 4])
            with cc1:
                if st.button("✅ Confirm", key=f"swap_confirm__{slot_idx}") and new_movie:
                    old = st.session_state["taste_movies"][slot_idx]
                    st.session_state["taste_profile"].pop(old, None)
                    st.session_state["taste_movies"][slot_idx] = new_movie
                    st.session_state[swap_key] = False
                    st.rerun()
            with cc2:
                if st.button("✖ Cancel", key=f"swap_cancel__{slot_idx}"):
                    st.session_state[swap_key] = False
                    st.rerun()

    st.divider()

    # Profile summary
    if st.session_state["taste_profile"]:
        st.markdown("### 📋 Your Current Taste Profile")
        profile_data = [
            {"Movie": k, "Your Rating": f"{'⭐' * int(v)} ({v}/5)"}
            for k, v in st.session_state["taste_profile"].items()
        ]
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

        # Genre affinity chart
        genre_affinity = {}
        for title, rating in st.session_state["taste_profile"].items():
            r = movies_df[movies_df["title"] == title]
            if r.empty:
                continue
            for g in r.iloc[0]["genres"].split("|"):
                genre_affinity.setdefault(g, []).append(rating)

        if genre_affinity:
            aff_df = pd.DataFrame([
                {"Genre": g, "Avg Rating": round(sum(v)/len(v), 2)}
                for g, v in genre_affinity.items()
            ]).sort_values("Avg Rating", ascending=False)
            fig = px.bar(aff_df, x="Genre", y="Avg Rating",
                         title="🎭 Your Genre Preferences",
                         color="Avg Rating", color_continuous_scale="Reds",
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🗑 Clear Taste Profile"):
            st.session_state["taste_profile"]    = {}
            st.session_state["taste_initialized"] = False
            st.rerun()
    else:
        st.info("Rate at least one movie above to see your taste profile.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>"
    "🎬 CineMatch AI · MovieLens 1M · TF-IDF · SVD · KNN · Hybrid"
    "</div>",
    unsafe_allow_html=True
)
