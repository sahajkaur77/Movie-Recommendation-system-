#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import ast
from nltk.stem.porter import PorterStemmer

# --- Data Loading ---
try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
except FileNotFoundError:
    print("Error: tmdb_5000_movies.csv and/or tmdb_5000_credits.csv not found.")
    print("Please download the dataset from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata and place the files in the same directory as this script.")
    exit()

# --- Data Preprocessing ---
movies = movies.merge(credits, on='movie_id')
movies.rename(columns={'title_x': 'title'}, inplace=True)
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# --- Helper Functions for Data Transformation ---
def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except (ValueError, SyntaxError):
        pass
    return L

def convert3(obj):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except (ValueError, SyntaxError):
        pass
    return L

def fetch_director(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except (ValueError, SyntaxError):
        pass
    return L

# --- Applying Transformations ---
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# --- Feature Engineering ---
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# --- Final DataFrame ---
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# --- Stemming ---
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# --- Vectorization and Similarity ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# --- Save Model Files ---
os.makedirs('model', exist_ok=True)
pickle.dump(new_df, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("Model files 'movie_list.pkl' and 'similarity.pkl' generated successfully in the 'model' directory.")