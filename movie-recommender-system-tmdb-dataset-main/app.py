import pickle
import streamlit as st
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Application started")

try:
    logger.info("Starting the application")

    def fetch_poster(movie_id):
        logger.info(f"Fetching poster for movie_id: {movie_id}")
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
        try:
            data = requests.get(url)
            data = data.json()
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            logger.info(f"Successfully fetched poster for movie_id: {movie_id}")
            return full_path
        except Exception as e:
            logger.error(f"Error fetching poster for movie_id: {movie_id}: {e}")
            return "https://via.placeholder.com/500x750.png?text=No+Poster+Found"


    def recommend(movie):
        logger.info(f"Getting recommendations for movie: {movie}")
        try:
            index = movies[movies['title'] == movie].index[0]
            distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
            recommended_movie_names = []
            recommended_movie_posters = []
            for i in distances[1:6]:
                # fetch the movie poster
                movie_id = movies.iloc[i[0]].movie_id
                recommended_movie_posters.append(fetch_poster(movie_id))
                recommended_movie_names.append(movies.iloc[i[0]].title)
            logger.info(f"Successfully got recommendations for movie: {movie}")
            return recommended_movie_names,recommended_movie_posters
        except Exception as e:
            logger.error(f"Error getting recommendations for movie: {movie}: {e}")
            return [], []

    st.header('Movie Recommender System')
    logger.info("Loading pickle files")
    try:
        movies = pickle.load(open('model/movie_list.pkl','rb'))
        similarity = pickle.load(open('model/similarity.pkl','rb'))
        logger.info("Successfully loaded pickle files")
    except Exception as e:
        logger.error(f"Error loading pickle files: {e}")
        st.error("Error loading model files. Please check the logs.")
        st.stop()


    movie_list = movies['title'].values
    logger.info("Creating selectbox")
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
    logger.info(f"Selected movie: {selected_movie}")


    logger.info("Creating button")
    if st.button('Show Recommendation'):
        logger.info("Button clicked")
        recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
        if recommended_movie_names:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])
        else:
            st.error("Could not get recommendations for the selected movie.")

except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")
    st.error("An unexpected error occurred. Please check the logs.")

if __name__ == '__main__':
    st.set_option('server.enableCORS', True)
    st.set_option('server.port', 8888)
