import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


merged_movies_df = pd.read_csv('movies_with_embeddings.csv')

# convert the saved string back to a NumPy array. Previously generating an error when 
def string_to_array(s):
    return np.array([float(item) for item in s.replace('[', '').replace(']', '').replace('\n', '').split(',')])

# Convert the 'embedding' column back to NumPy arrays
merged_movies_df['embedding'] = merged_movies_df['embedding'].apply(string_to_array)


st.title(':female-student: SCB\'s Movie Recommender :movie_camera:')

selected_movie = st.selectbox('Select a movie:', merged_movies_df['title'].tolist())

def format_imdb_url(imdb_id):
    return f"[IMDb](https://www.imdb.com/title/tt{imdb_id})"

def format_tmdb_url(tmdb_id):
    return f"[TMDb](https://www.themoviedb.org/movie/{tmdb_id})"

def recommend_similar_movies(movie_title, movies_df, top_n=5):
    target_embedding = movies_df[movies_df['title'] == movie_title]['embedding'].iloc[0]
    similarities = cosine_similarity([target_embedding], np.vstack(movies_df['embedding'].values))[0]
    top_indices = np.argsort(similarities)[-top_n - 1:][::-1][1:]  # Exclude the movie itself
    recommended_movies = movies_df.iloc[top_indices]
    return recommended_movies



if st.button('Recommend Similar Movies'):
    recommendations = recommend_similar_movies(selected_movie, merged_movies_df)

    for _, row in recommendations.iterrows():
        st.subheader(row['title'])
        st.markdown(f"**IMDb ID:** {format_imdb_url(row['imdbId'])} | **TMDb ID:** {format_tmdb_url(row['tmdbId'])}")
        st.write(f"**Rating:** {row['rating']:.2f}")
        st.write(f"**Year:** {int(row['year'])}")
        st.write(f"**Genre:** {row['genres']}")
        st.caption(f"**Embeddings:** {row['embedding']}")

