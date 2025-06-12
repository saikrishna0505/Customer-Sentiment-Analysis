from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')

# Initialize Flask App
app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.strip()
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    return ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])

# API Route for Sentiment Prediction
@app.route("/predict", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided."}), 400
        
        text = data['text']
        processed = clean_text(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        
        return jsonify({
            "input_text": text,
            "cleaned_text": processed,
            "predicted_sentiment": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

