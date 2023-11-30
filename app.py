from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz  # Import fuzzywuzzy for string matching

app = Flask(__name__)

# Load the MovieLens dataset
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Merge the datasets
movie_ratings = pd.merge(movies, ratings)

# Create a TF-IDF vectorizer for movie titles and genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['title'] + ' ' + movies['genres'])

# Compute the cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []  # Initialize an empty list for recommendations
    error_message = ""  # Initialize an empty error message
    if request.method == 'POST':
        # Get the user's input movie title
        movie_title = request.form['movie_title']

        # Check if the movie title exists in the dataset
        if movie_title in movies['title'].values:
            # Get recommendations based on the user's input
            recommendations = get_recommendations(movie_title)
        else:
            # Use fuzzy string matching to find similar movie titles
            similar_titles = find_similar_titles(movie_title)

            if similar_titles:
                recommendations = get_recommendations(similar_titles[0])  # Use the most similar title
            else:
                # Set an error message if no similar titles are found
                error_message = "Movie not found or similar in the dataset"

    return render_template('index.html', recommendations=recommendations, error_message=error_message)

def get_recommendations(title):
    # Find the index of the movie in the dataset
    idx = movies[movies['title'] == title].index[0]

    # Get the pairwise cosine similarity scores of the movie with all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 recommended movies
    return movies['title'].iloc[movie_indices].tolist()

def find_similar_titles(input_title, threshold=70):
    similar_titles = []
    for title in movies['title']:
        # Use fuzzywuzzy's token_set_ratio to measure string similarity
        similarity = fuzz.token_set_ratio(input_title.lower(), title.lower())
        if similarity >= threshold:
            similar_titles.append(title)

    return similar_titles

if __name__ == '__main__':
    app.run(debug=True)
