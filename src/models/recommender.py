"""
Movie Recommendation System - Production Module
Item-Based Collaborative Filtering Implementation
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


class MovieRecommender:
    """
    Production-ready Movie Recommendation System
    Uses Item-Based Collaborative Filtering with Cosine Similarity
    """
    
    def __init__(self):
        """Initialize the recommender system"""
        self.user_item_matrix = None
        self.item_similarity_df = None
        self.movies_df = None
        self.is_trained = False
        self.training_info = {
            'trained_at': None,
            'num_users': 0,
            'num_movies': 0,
            'num_ratings': 0,
            'sparsity': 0.0,
            'model_version': '1.0'
        }
    
    def train(self, ratings_df, movies_df, verbose=True):
        """Train the recommendation model"""
        if verbose:
            print("Training Movie Recommender...")
        
        self.movies_df = movies_df.copy()
        ratings_df = ratings_df.drop_duplicates(['userId', 'movieId'], keep='last')
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Calculate sparsity
        num_ratings = (self.user_item_matrix > 0).sum().sum()
        total_cells = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        sparsity = (1 - num_ratings / total_cells) * 100
        
        # Calculate item similarity
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        # Update training info
        self.training_info = {
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_users': self.user_item_matrix.shape[0],
            'num_movies': self.user_item_matrix.shape[1],
            'num_ratings': len(ratings_df),
            'sparsity': sparsity,
            'avg_rating': float(ratings_df['rating'].mean()),
            'model_version': '1.0',
            'algorithm': 'Item-Based Collaborative Filtering',
            'similarity_metric': 'Cosine Similarity'
        }
        
        self.is_trained = True
        
        if verbose:
            print(f"✅ Model trained: {self.training_info['num_movies']} movies, {self.training_info['num_users']} users")
        
        return self.training_info
    
    def recommend(self, user_ratings, n_recommendations=10, min_similarity=0.0):
        """Generate movie recommendations"""
        if not self.is_trained:
            raise Exception("Model not trained! Call train() first.")
        
        if not user_ratings:
            raise ValueError("user_ratings cannot be empty")
        
        scores = {}
        
        for movie_id, rating in user_ratings.items():
            if movie_id not in self.item_similarity_df.columns:
                continue
            
            similar_movies = self.item_similarity_df[movie_id]
            
            for other_movie_id, similarity in similar_movies.items():
                if other_movie_id in user_ratings:
                    continue
                
                if similarity <= min_similarity:
                    continue
                
                if other_movie_id not in scores:
                    scores[other_movie_id] = 0
                scores[other_movie_id] += similarity * rating
        
        if not scores:
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'score'])
        
        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        recommended_ids = [movie_id for movie_id, score in top_movies]
        
        recommendations = self.movies_df[self.movies_df['movieId'].isin(recommended_ids)].copy()
        score_dict = dict(top_movies)
        recommendations['score'] = recommendations['movieId'].map(score_dict)
        recommendations = recommendations.sort_values('score', ascending=False)
        
        return recommendations[['movieId', 'title', 'genres', 'score']].reset_index(drop=True)
    
    def get_similar_movies(self, movie_id, n_similar=10, min_similarity=0.3):
        """Find movies similar to a given movie"""
        if not self.is_trained:
            raise Exception("Model not trained!")
        
        if movie_id not in self.item_similarity_df.columns:
            raise ValueError(f"Movie ID {movie_id} not found")
        
        similarities = self.item_similarity_df[movie_id]
        similar = similarities[similarities >= min_similarity].sort_values(ascending=False)[1:n_similar+1]
        
        similar_movies = self.movies_df[self.movies_df['movieId'].isin(similar.index)].copy()
        similar_movies['similarity'] = similar_movies['movieId'].map(similar)
        similar_movies = similar_movies.sort_values('similarity', ascending=False)
        
        return similar_movies[['movieId', 'title', 'genres', 'similarity']].reset_index(drop=True)
    
    def get_info(self):
        """Get model information"""
        if not self.is_trained:
            return {"status": "Model not trained"}
        return self.training_info
    
    def save(self, filepath='models/recommender_model.pkl'):
        """Save model to disk"""
        if not self.is_trained:
            raise Exception("Cannot save untrained model!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'item_similarity_df': self.item_similarity_df,
            'movies_df': self.movies_df,
            'training_info': self.training_info
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        return filepath
    
    def load(self, filepath='models/recommender_model.pkl'):
        """Load model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_similarity_df = model_data['item_similarity_df']
        self.movies_df = model_data['movies_df']
        self.training_info = model_data['training_info']
        self.is_trained = True
        
        return self
