from flask import Flask, request, render_template
import pickle as pk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
model = pk.load(open("C:/Users/USER/Desktop/sulap/sulap/model.pkl", 'rb'))
vectorizer = pk.load(open("C:/Users/USER/Desktop/sulap/sulap/scaler.pkl", 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index1.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    # Vectorize the review
    review_vectorized = vectorizer.transform([review])

    # Predict sentiment
    result = model.predict(review_vectorized)

    # Display result
    if result[0] == 0:
        sentiment = 'Negative Review'
    else:
        sentiment = 'Positive Review'

    return render_template('result1.html', review=review, sentiment=sentiment)

if __name__== '__main__':
    app.run(debug=True)
