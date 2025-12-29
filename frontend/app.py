"""
Movie Recommendation System - Frontend Interface
- Feature: Evolving Recommendations Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.models.recommender import MovieRecommender
except ImportError:
    st.error("Could not import MovieRecommender. Make sure you're running from project root.")

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .recommendation-score {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .genre-tag {
        background-color: #f0f2f6;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.85rem;
        margin-right: 0.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()

init_session_state()

# Load model and data
@st.cache_resource(show_spinner=False)
def load_model_and_data():
    """Load the recommendation model and movie data"""
    try:
        # Try loading from pickle files first
        ratings_df = pd.read_pickle('../data/ratings_sample.pkl')
        movies_df = pd.read_pickle('../data/movies_sample.pkl')
        data_source = "Local files"
    except:
        try:
            # Fallback to CSV
            ratings_df = pd.read_csv('../data/ratings_sample.csv')
            movies_df = pd.read_csv('../data/movies_sample.csv')
            data_source = "CSV files"
        except:
            # Last resort: Load from BigQuery
            from google.cloud import bigquery
            client = bigquery.Client()
            
            ratings_df = client.query("""
                SELECT userId, movieId, rating
                FROM `master-ai-cloud.MoviePlatform.ratings`
                LIMIT 50000
            """).to_dataframe()
            
            movies_df = client.query("""
                SELECT movieId, title, genres
                FROM `master-ai-cloud.MoviePlatform.movies`
            """).to_dataframe()
            
            data_source = "BigQuery"
    
    # Train model
    recommender = MovieRecommender()
    recommender.train(ratings_df, movies_df, verbose=False)
    
    return recommender, movies_df, data_source

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover movies you\'ll love • Watch your recommendations evolve</p>', unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading recommendation engine..."):
            try:
                recommender, movies_df, data_source = load_model_and_data()
                st.session_state.recommender = recommender
                st.session_state.movies_df = movies_df
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded from {data_source}!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 Your Profile")
        num_ratings = len(st.session_state.user_ratings)
        st.metric("Movies Rated", num_ratings)
        st.metric("Current Step", st.session_state.step + 1)
        
        st.markdown("---")
        if st.session_state.user_ratings:
            st.markdown("### ⭐ Recent Ratings")
            for movie_id, rating in list(st.session_state.user_ratings.items())[-5:]:
                movie = st.session_state.movies_df[st.session_state.movies_df['movieId'] == movie_id]
                if not movie.empty:
                    title = movie['title'].values[0][:30]
                    st.write(f"{'⭐' * int(rating)} {title}")
        
        st.markdown("---")
        if st.button("🔄 Start Over"):
            st.session_state.user_ratings = {}
            st.session_state.step = 0
            st.rerun()
    
    # Main content
    recommender = st.session_state.recommender
    movies_df = st.session_state.movies_df
    num_ratings = len(st.session_state.user_ratings)
    
    st.markdown(f'<div class="step-badge">Step {st.session_state.step + 1}</div>', unsafe_allow_html=True)
    
    if num_ratings < 3:
        st.markdown("## 🎯 Let's Get Started!")
        st.write("Rate a few movies to get personalized recommendations.")
        
        # Show random movies
        if 'initial_movies' not in st.session_state:
            st.session_state.initial_movies = movies_df.sample(min(20, len(movies_df)), random_state=42)
        
        for _, movie in st.session_state.initial_movies.head(10).iterrows():
            if movie['movieId'] not in st.session_state.user_ratings:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{movie['title']}**")
                        st.caption(f"Genres: {movie['genres']}")
                    with col2:
                        rating = st.select_slider(
                            "Rate",
                            options=[1.0, 2.0, 3.0, 4.0, 5.0],
                            value=3.0,
                            key=f"rate_{movie['movieId']}"
                        )
                        if st.button("✓", key=f"btn_{movie['movieId']}"):
                            st.session_state.user_ratings[movie['movieId']] = rating
                            st.session_state.step += 1
                            st.rerun()
                    st.markdown("---")
    else:
        st.markdown("## 🎬 Your Recommendations")
        
        try:
            recs = recommender.recommend(st.session_state.user_ratings, n_recommendations=10)
            
            if recs.empty:
                st.warning("No recommendations yet. Rate more movies!")
            else:
                st.success(f"Based on your {num_ratings} ratings:")
                
                for idx, rec in recs.head(5).iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{idx+1}. {rec['title']}**")
                            st.caption(f"{rec['genres']} | Score: {rec['score']:.2f}")
                        with col2:
                            rating = st.select_slider(
                                "Rate",
                                options=[1.0, 2.0, 3.0, 4.0, 5.0],
                                value=3.0,
                                key=f"rec_rate_{rec['movieId']}"
                            )
                            if st.button("✓", key=f"rec_btn_{rec['movieId']}"):
                                st.session_state.user_ratings[rec['movieId']] = rating
                                st.session_state.step += 1
                                st.rerun()
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Stats
    if num_ratings >= 3:
        st.markdown("## 📊 Your Journey")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ratings", num_ratings)
        with col2:
            avg = np.mean(list(st.session_state.user_ratings.values()))
            st.metric("Average Rating", f"{avg:.1f}⭐")
        with col3:
            progress = min(num_ratings * 10, 100)
            st.metric("Discovery", f"{progress}%")

if __name__ == "__main__":
    main()
