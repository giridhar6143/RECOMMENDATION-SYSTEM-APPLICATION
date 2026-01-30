import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown(
    "**Collaborative Filtering (User-Based)**  \n"
    "Optimized with Sparse Matrices & Caching"
)

# ------------------------------------
# Load Data
# ------------------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(r"D:\RECOMMENDATION-SYSTEM-APPLICATION\ratings.csv")
    movies = pd.read_csv(r"D:\RECOMMENDATION-SYSTEM-APPLICATION\movies.csv")
    return ratings, movies

ratings_df, movies_df = load_data()

# ------------------------------------
# Dataset Preview
# ------------------------------------
with st.expander("📊 Dataset Preview"):
    st.write("Ratings Dataset", ratings_df.head())
    st.write("Movies Dataset", movies_df.head())

# ------------------------------------
# Train-Test Split
# ------------------------------------
train_df, test_df = train_test_split(
    ratings_df,
    test_size=0.2,
    random_state=42
)

# ------------------------------------
# User–Item Matrix
# ------------------------------------
user_item_matrix = train_df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

user_item_filled = user_item_matrix.fillna(0)

# ------------------------------------
# User Similarity (Cosine)
# ------------------------------------
@st.cache_resource
def compute_similarity(matrix):
    sparse_matrix = csr_matrix(matrix.values)
    similarity = cosine_similarity(sparse_matrix, dense_output=False)
    return pd.DataFrame(
        similarity.toarray(),
        index=matrix.index,
        columns=matrix.index
    )

user_similarity_df = compute_similarity(user_item_filled)

# ------------------------------------
# Predict Rating Function
# ------------------------------------
def predict_rating(user_id, movie_id, k=10):
    if movie_id not in user_item_matrix.columns or user_id not in user_item_matrix.index:
        return np.nan
    
    sim_scores = user_similarity_df[user_id]
    movie_ratings = user_item_matrix[movie_id]
    valid_users = movie_ratings.dropna().index
    
    if valid_users.empty:
        return np.nan
    
    top_users = sim_scores.loc[valid_users].nlargest(k)
    
    if top_users.empty:
        return np.nan
    
    return np.dot(movie_ratings.loc[top_users.index], top_users.values) / top_users.sum()

# ------------------------------------
# Model Evaluation
# ------------------------------------
def evaluate_model(test_df, k=10):
    preds = [
        predict_rating(row['userId'], row['movieId'], k)
        for _, row in test_df.iterrows()
    ]
    mask = ~pd.isna(preds)
    y_true = test_df.loc[mask, 'rating']
    y_pred = np.array(preds)[mask]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

rmse, mae = evaluate_model(test_df, k=10)

# ------------------------------------
# Display Metrics
# ------------------------------------
st.subheader("📈 Model Evaluation")

col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.4f}")
col2.metric("MAE", f"{mae:.4f}")

# ------------------------------------
# Recommendation Function
# ------------------------------------
def recommend_movies(user_id, n=10, k=10):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['title', 'genres', 'predicted_rating'])
    
    user_rated = ratings_df[ratings_df['userId'] == user_id]['movieId']
    all_movies = movies_df['movieId']
    
    recommendations = []
    for movie_id in all_movies:
        if movie_id not in user_rated.values:
            pred = predict_rating(user_id, movie_id, k)
            if not np.isnan(pred):
                recommendations.append((movie_id, pred))
    
    if not recommendations:
        return pd.DataFrame(columns=['title', 'genres', 'predicted_rating'])
    
    rec_df = pd.DataFrame(recommendations, columns=['movieId', 'predicted_rating'])
    return rec_df.sort_values('predicted_rating', ascending=False).head(n).merge(movies_df, on='movieId')

# ------------------------------------
# User Interaction
# ------------------------------------
st.subheader("🎯 Get Recommendations")

user_ids = sorted(ratings_df['userId'].unique())
selected_user = st.selectbox("Select User ID", user_ids)

top_n = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("🔍 Recommend"):
    recommendations = recommend_movies(selected_user, top_n, k=10)
    st.success(f"Top {top_n} Movies for User {selected_user}")
    st.dataframe(
        recommendations[['title', 'genres', 'predicted_rating']],
        use_container_width=True
    )

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.markdown("Collaborative Filtering using Cosine Similarity (Optimized)")
