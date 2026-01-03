"""
FastAPI Backend for Movie Recommendation System
Integrates the ML model and Groq LLM from the notebook
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import pickle
import pandas as pd
import numpy as np
from groq import Groq
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Movie Recommendation API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://54b7fa68cd4d3403-dot-europe-west1.notebooks.googleusercontent.com",
        "https://54b7fa68cd4d3403-8000-dot-europe-west1.notebooks.googleusercontent.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============================================================================
# DATA MODELS
# ============================================================================

class RatingInput(BaseModel):
    """User rating input"""
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)
    
    @validator('rating')
    def rating_must_be_valid(cls, v):
        if v < 0.5 or v > 5.0:
            raise ValueError('Rating must be between 0.5 and 5.0')
        return v

class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    ratings: Dict[int, float]
    num_recommendations: int = Field(default=10, ge=1, le=50)
    
    @validator('ratings')
    def ratings_not_empty(cls, v):
        if not v:
            raise ValueError('Ratings cannot be empty')
        return v
    
    @validator('ratings')
    def validate_rating_values(cls, v):
        for movie_id, rating in v.items():
            if rating < 0.5 or rating > 5.0:
                raise ValueError(f'Rating for movie {movie_id} must be between 0.5 and 5.0')
        return v

class Movie(BaseModel):
    """Movie information"""
    id: int
    title: str
    genres: str

class RecommendationResponse(BaseModel):
    """Recommendation response"""
    movies: List[Dict]
    message: Optional[str] = None

# ============================================================================
# MOVIE RECOMMENDER CLASS
# ============================================================================

class MovieRecommenderAPI:
    """ML-based movie recommendation engine"""
    
    def __init__(self, model_path: str):
        logger.info("Loading recommendation model...")
        
        try:
            with open(model_path, 'rb') as f:
                components = pickle.load(f)
            
            self.user_item_matrix = components['user_item_matrix']
            self.item_similarity_df = components['item_similarity_df']
            self.movies_df = components['movies_df']
            self.metrics = components['metrics']
            
            logger.info(f"Model loaded: {len(self.movies_df):,} movies")
            logger.info(f"Hit Rate: {self.metrics['hit_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def validate_movie_ids(self, movie_ids: List[int]) -> bool:
        """Validate that movie IDs exist in the dataset"""
        valid_ids = set(self.movies_df['movieId'].values)
        return all(mid in valid_ids for mid in movie_ids)
    
    def get_recommendations(self, user_ratings: Dict[int, float], n: int = 10):
        """Generate recommendations using collaborative filtering"""
        scores = {}
        
        # Validate movie IDs
        invalid_ids = [mid for mid in user_ratings.keys() 
                      if mid not in self.movies_df['movieId'].values]
        if invalid_ids:
            logger.warning(f"Invalid movie IDs provided: {invalid_ids}")
        
        for movie_id, rating in user_ratings.items():
            if movie_id not in self.item_similarity_df.columns:
                continue
            
            similar_movies = self.item_similarity_df[movie_id]
            
            for other_movie_id, similarity in similar_movies.items():
                if other_movie_id in user_ratings or similarity <= 0:
                    continue
                
                if other_movie_id not in scores:
                    scores[other_movie_id] = 0
                scores[other_movie_id] += similarity * rating
        
        if not scores:
            return pd.DataFrame()
        
        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recommended_ids = [mid for mid, _ in top_movies]
        
        recommendations = self.movies_df[
            self.movies_df['movieId'].isin(recommended_ids)
        ].copy()
        
        score_dict = dict(top_movies)
        recommendations['score'] = recommendations['movieId'].map(score_dict)
        recommendations = recommendations.sort_values('score', ascending=False)
        
        return recommendations[['movieId', 'title', 'genres', 'score']]
    
    def get_all_movies(self, limit: int = 100):
        """Get a sample of movies for initial display"""
        # Return with consistent 'id' property name for frontend
        movies = self.movies_df.head(limit)[['movieId', 'title', 'genres']].to_dict('records')
        # Rename movieId to id for frontend consistency
        return [{'id': m['movieId'], 'title': m['title'], 'genres': m['genres']} for m in movies]
    
    def get_movie_info(self, movie_id: int):
        """Get information about a specific movie"""
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie.empty:
            return None
        movie_dict = movie.iloc[0].to_dict()
        # Rename movieId to id for consistency
        movie_dict['id'] = movie_dict.pop('movieId')
        return movie_dict

# ============================================================================
# CINEPHILE ASSISTANT (OPTIONAL - FOR LLM ENHANCEMENT)
# ============================================================================

class CinephileAssistant:
    """Optional LLM wrapper for conversational recommendations"""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are a passionate movie recommendation assistant!
        
        RULES:
        - Only recommend movies from the provided list
        - Be warm and enthusiastic but concise
        - Use 2-3 emojis per message
        - Keep responses under 150 words
        """
    
    def generate_message(self, recommendations_df, user_ratings: Dict[int, float]):
        """Generate a friendly message about recommendations"""
        try:
            # Format recommendations
            recs_text = "\n".join([
                f"{i+1}. {row['title']} ({row['genres']}) - Score: {row['score']:.2f}"
                for i, (_, row) in enumerate(recommendations_df.head(5).iterrows())
            ])
            
            prompt = f"""Based on the user's ratings, here are the top recommendations:

{recs_text}

Write a brief, enthusiastic message (2-3 sentences) introducing these recommendations."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating LLM message: {e}")
            return "Here are your personalized recommendations!"

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

def find_model_path():
    """Try to locate the model file in common locations"""
    # Check environment variable first
    env_path = os.getenv('MODEL_PATH')
    if env_path and Path(env_path).exists():
        logger.info(f"Found model at env MODEL_PATH: {env_path}")
        return env_path
    
    possible_paths = [
        "../../models/saved_models/recommender_components.pkl",
        "../models/saved_models/recommender_components.pkl",
        "./models/saved_models/recommender_components.pkl",
        "models/saved_models/recommender_components.pkl",
        "recommender_components.pkl",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"Found model at: {path}")
            return path
    
    logger.error("Model file not found in any expected location!")
    logger.error("Tried these paths:")
    for path in possible_paths:
        logger.error(f"  - {path}")
    logger.error("Set MODEL_PATH environment variable or update app.py")
    return None

MODEL_PATH = find_model_path()
recommender = None

if MODEL_PATH:
    try:
        recommender = MovieRecommenderAPI(MODEL_PATH)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        logger.warning("The API will start but recommendations won't work until model is loaded")
else:
    logger.warning("Model path not set. Recommendations will not work.")

# Optional: Initialize LLM assistant if API key is available
assistant = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    try:
        assistant = CinephileAssistant(GROQ_API_KEY)
        logger.info("LLM assistant enabled")
    except Exception as e:
        logger.warning(f"LLM assistant not available: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the frontend HTML directly"""
    possible_paths = [
        Path("../frontend/demo_integrated.html"),
        Path("../../frontend/demo_integrated.html"),
        Path("/home/jupyter/Chiheb_Ramy/GCP-Personalized-Movie-Recommendation-System/frontend/demo_integrated.html"),
        Path.home() / "Chiheb_Ramy/GCP-Personalized-Movie-Recommendation-System/frontend/demo_integrated.html"
    ]
    
    frontend_path = None
    for path in possible_paths:
        if path.exists():
            frontend_path = path
            logger.info(f"Found frontend at: {path}")
            break
    
    if frontend_path and frontend_path.exists():
        from fastapi.responses import HTMLResponse
        with open(frontend_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    
    logger.warning(f"Frontend not found. Tried: {[str(p) for p in possible_paths]}")
    
    return {
        "status": "online",
        "message": "Movie Recommendation API is running!",
        "model_loaded": recommender is not None,
        "llm_enabled": assistant is not None,
        "note": "Frontend not found. Please check the path."
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Movie Recommendation API is running!",
        "model_loaded": recommender is not None,
        "llm_enabled": assistant is not None
    }

@app.get("/api/movies")
async def get_movies(limit: int = 50):
    """Get a sample of movies for initial display"""
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
    
    movies = recommender.get_all_movies(limit)
    return {"movies": movies, "total": len(movies)}

@app.get("/api/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get details about a specific movie"""
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    movie = recommender.get_movie_info(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return movie

@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized movie recommendations"""
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.ratings:
        raise HTTPException(status_code=400, detail="No ratings provided")
    
    # Log request for debugging
    logger.info(f"Recommendation request: {len(request.ratings)} ratings, {request.num_recommendations} recommendations")
    
    # Get recommendations from ML model
    recommendations_df = recommender.get_recommendations(
        request.ratings,
        request.num_recommendations
    )
    
    if recommendations_df.empty:
        return {
            "movies": [],
            "message": "No recommendations found. Try rating more movies!"
        }
    
    # Convert to list of dicts with consistent 'id' property
    recommendations = [
        {
            "id": int(row['movieId']),
            "title": row['title'],
            "genres": row['genres'],
            "score": float(row['score'])
        }
        for _, row in recommendations_df.iterrows()
    ]
    
    # Optional: Generate LLM message
    message = None
    if assistant:
        try:
            message = assistant.generate_message(recommendations_df, request.ratings)
        except Exception as e:
            logger.error(f"Error generating LLM message: {e}")
            message = "Here are your personalized recommendations!"
    
    return {
        "movies": recommendations,
        "message": message,
        "total_ratings": len(request.ratings)
    }

@app.post("/api/rate")
async def rate_movie(rating: RatingInput):
    """Record a movie rating (for future use)"""
    # In a real app, you'd store this in a database
    logger.info(f"Rating recorded: Movie {rating.movie_id} = {rating.rating} stars")
    return {
        "success": True,
        "movie_id": rating.movie_id,
        "rating": rating.rating,
        "message": "Rating recorded!"
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 80)
    logger.info("MOVIE RECOMMENDATION API")
    logger.info("=" * 80)
    logger.info("Starting server on http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")
    if not recommender:
        logger.warning("WARNING: Model not loaded - recommendations won't work!")
        logger.warning("Update MODEL_PATH in app.py with your model file location")
    logger.info("=" * 80)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)