# 🎬 Movie Recommendation System

## Project Overview
A content-based movie recommendation system that suggests movies similar to a selected movie.

### Technique
- MovieLens Latest Small dataset
- Genre-based features
- TF-IDF vectorization
- Cosine similarity
- Streamlit interface

The MovieLens small dataset contains about 100,000 ratings and data for about 9,000 movies. GroupLens provides it specifically as an educational/development dataset. The application downloads `movies.csv` automatically when first launched.

## Run

```bash
cd 01_Movie_Recommendation_System
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the Streamlit application.

## How It Works

1. Download/load MovieLens movie metadata.
2. Convert the pipe-separated genres into text features.
3. Convert genre text into TF-IDF vectors.
4. Calculate cosine similarity between movies.
5. Return the most similar movies.

## Example

Selecting:

```text
Toy Story (1995)
```

will return movies with similar genre profiles.

## Dataset Source

MovieLens Latest Small — GroupLens Research.

Official dataset information:
https://grouplens.org/datasets/movielens/latest/

## Limitations

This is a content-based prototype. It recommends movies primarily from genre similarity and does not yet model individual user-rating history.
