
# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Streamlit is used to create the web application UI
import streamlit as st

# Pandas is used for data handling and dataframe operations
import pandas as pd

# Joblib is used to load saved machine learning files
import joblib

# linear_kernel is used to calculate cosine similarity
from sklearn.metrics.pairwise import linear_kernel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

# Configure Streamlit page settings
st.set_page_config(

    # Browser tab title
    page_title="Movie Recommendation System",

    # Wide layout for better UI spacing
    layout="wide"
)


# ============================================================
# CUSTOM CSS + FONT
# ============================================================

# Add custom CSS styling for beautiful UI
st.markdown("""
<style>

/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Apply font globally */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main app background */
.stApp {
    background-color: #141414;
    color: white;
}

/* Movie recommendation card styling */
.movie-card {
    background-color: #1f1f1f;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    transition: 0.3s;
    min-height: 320px;
}

/* Hover effect on movie cards */
.movie-card:hover {
    transform: scale(1.03);
    background-color: #2a2a2a;
}

/* Movie title styling */
.movie-title {
    color: #E50914;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Movie text styling */
.movie-text {
    color: white;
    font-size: 15px;
    margin-bottom: 8px;
}

/* Search box styling */
.stTextInput input {
    background-color: #2a2a2a;
    color: white;
    font-size: 16px;
    border-radius: 10px;
}

/* Button styling */
.stButton button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 600;
}

/* Button hover effect */
.stButton button:hover {
    background-color: #ff1f1f;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

# Load processed movie dataset
# This file contains cleaned movie data
data = joblib.load(
    'models/movie_data.pkl'
)

# Load saved TF-IDF vectorizer
# This converts text into numerical vectors
tfidf = joblib.load(
    'models/tfidf_vectorizer.pkl'
)

# Load TF-IDF matrix
# This matrix stores vectorized movie features
tfidf_matrix = joblib.load(
    'models/tfidf_matrix.pkl'
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

# Function to recommend similar movies
# movie  -> movie name entered by user
# top_n  -> number of recommendations

def recommend_movies(movie, top_n=10):

    # Convert movie name into lowercase
    # This improves matching accuracy
    movie = movie.lower()

    # Find movie matching user input
    # str.contains searches partial movie names
    m_movie = data[
        data['title'].str.lower().str.contains(
            movie,
            na=False
        )
    ]

    # If movie is not found
    # return empty list
    if m_movie.empty:
        return []

    # Get index of selected movie
    m_index = m_movie.index[0]

    # Calculate cosine similarity
    # Compare selected movie with all movies
    similarity_score = linear_kernel(
        tfidf_matrix[m_index:m_index+1],
        tfidf_matrix
    ).flatten()

    # Sort similarity scores in descending order
    similar_movie = similarity_score.argsort()[::-1]

    # Create set to remove duplicate movie names
    seen_movies = set()

    # Store final recommendations
    recommendations = []

    # Loop through similar movies
    for idx in similar_movie:

        # Get movie title
        title = data.iloc[idx]['title']

        # Skip duplicate movie titles
        if title in seen_movies:
            continue

        # Add movie into set
        seen_movies.add(title)

        # Append movie details into recommendation list
        recommendations.append({

            # Convert title into title case
            "Movie":
            title.title(),

            # Genre of movie
            "Genre":
            data.iloc[idx]['genre'],

            # Director name
            "Director":
            data.iloc[idx]['director'],

            # Audience score rounded to 2 decimal places
            "Audience Score":
            round(
                data.iloc[idx]['audienceScore'],
                2
            ),

            # Tomato meter score rounded to 2 decimals
            "Tomato Meter":
            round(
                data.iloc[idx]['tomatoMeter'],
                2
            ),

            # Similarity score rounded to 4 decimals
            "Similarity":
            round(
                similarity_score[idx],
                4
            )
        })

        # Stop after reaching top_n recommendations
        if len(recommendations) == top_n:
            break

    # Return final recommendations
    return recommendations


# ============================================================
# RED HEADER SECTION
# ============================================================

# Create beautiful red header section
st.markdown("""
<div style="
background-color:#E50914;
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:30px;
">

<h1 style="
color:white;
font-size:55px;
margin:0;
">
🎬 MOVIE RECOMMENDATION SYSTEM
</h1>

<p style="
color:white;
font-size:20px;
margin-top:10px;
">
By Vedika Bengani
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SEARCH BAR
# ============================================================

# Create movie search input box
movie_name = st.text_input(

    # Search bar label
    "🔍 Search Your Favorite Movie"
)


# ============================================================
# RECOMMENDATION SLIDER
# ============================================================

# Slider for selecting recommendation count
top_n = st.slider(

    # Slider title
    "🎥 Number of Recommendations",

    # Minimum value
    5,

    # Maximum value
    25,

    # Default value
    10
)


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

# Run recommendation system after button click
if st.button("🎬 Recommend Movies"):

    # Call recommendation function
    recommendations = recommend_movies(
        movie_name,
        top_n
    )

    # If movie not found
    if not recommendations:

        # Show error message
        st.error(
            "❌ Movie not found."
        )

    else:

        # Display recommendation heading
        st.subheader(
            "🍿 Recommended Movies"
        )

        # Create 3 columns layout
        cols = st.columns(3)

        # Loop through recommendations
        for i, movie in enumerate(recommendations):

            # Display movie in columns
            with cols[i % 3]:

                # Create movie card
                st.markdown(
                    f"""
                    <div class="movie-card">

                    <div class="movie-title">
                    {movie['Movie']}
                    </div>

                    <div class="movie-text">
                    🎭 Genre:
                    {movie['Genre']}
                    </div>

                    <div class="movie-text">
                    🎬 Director:
                    {movie['Director']}
                    </div>

                    <div class="movie-text">
                    ⭐ Audience Score:
                    {movie['Audience Score']}
                    </div>

                    <div class="movie-text">
                    🍅 Tomato Meter:
                    {movie['Tomato Meter']}
                    </div>

                    <div class="movie-text">
                    📊 Similarity:
                    {movie['Similarity']}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Display similarity progress bar
                st.progress(
                    min(movie['Similarity'], 1.0)
                )
# to run == streamlit run app/app.py