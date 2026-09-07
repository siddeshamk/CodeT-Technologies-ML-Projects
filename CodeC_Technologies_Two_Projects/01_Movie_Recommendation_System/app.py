import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).parent / "data"
MOVIES_FILE = DATA_DIR / "movies.csv"


@st.cache_data
def load_movies():
    if not MOVIES_FILE.exists():
        DATA_DIR.mkdir(exist_ok=True)
        with urlopen(DATA_URL, timeout=30) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = "ml-latest-small/movies.csv"
            with archive.open(member) as source:
                MOVIES_FILE.write_bytes(source.read())

    return pd.read_csv(MOVIES_FILE)


@st.cache_resource
def build_recommender(movies):
    data = movies.copy()
    data["genres_clean"] = data["genres"].fillna("").str.replace("|", " ", regex=False)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(data["genres_clean"])
    similarity = cosine_similarity(matrix)

    return data, similarity


def recommend(title, data, similarity, n=10):
    matches = data.index[data["title"].str.casefold() == title.casefold()].tolist()

    if not matches:
        partial = data[data["title"].str.contains(title, case=False, na=False)]
        if partial.empty:
            return pd.DataFrame(columns=["title", "genres"])
        index = partial.index[0]
    else:
        index = matches[0]

    scores = list(enumerate(similarity[index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    results = []
    for idx, score in scores[1:]:
        results.append({
            "title": data.loc[idx, "title"],
            "genres": data.loc[idx, "genres"],
            "similarity": round(float(score), 3),
        })
        if len(results) == n:
            break

    return pd.DataFrame(results)


st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("A content-based recommender using movie genres and TF-IDF cosine similarity.")

try:
    movies = load_movies()
    data, similarity = build_recommender(movies)

    selected = st.selectbox(
        "Choose a movie",
        data["title"].sort_values().tolist(),
        index=0,
    )
    count = st.slider("Number of recommendations", 5, 15, 10)

    if st.button("Recommend Movies", type="primary"):
        result = recommend(selected, data, similarity, count)
        st.subheader("Recommended Movies")
        if result.empty:
            st.warning("No matching movie was found.")
        else:
            st.dataframe(result, use_container_width=True, hide_index=True)

    st.caption(f"Dataset: {len(data):,} movies from MovieLens Latest Small.")
except Exception as exc:
    st.error(f"Could not load the MovieLens dataset: {exc}")
    st.info("Check your internet connection and try again.")
