from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# --- Robust NLTK Data Downloader ---
# Ensures the app has the necessary NLTK data when it starts.
print("Checking for NLTK data packages for the app...")
nltk_packages = ['stopwords', 'wordnet', 'omw-1.4']
for package in nltk_packages:
    try:
        nltk.data.find(f'corpora/{package}.zip')
    except LookupError:
        print(f"Downloading NLTK package for app: {package}")
        nltk.download(package)
print("NLTK packages are ready for the app.")


# --- Load Trained Model and Vectorizer ---
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("\n[ERROR] Model or vectorizer files not found.")
    print("Please run the 'train_model.py' script first to generate these files.")
    exit()

# --- Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)) # Ensure text is a string
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Define Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        processed_text = preprocess_text(review_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0] # Get the single prediction from the array

        # --- UPDATED: Map prediction index to sentiment label ---
        # Model outputs: 0=Negative, 1=Neutral, 2=Positive
        if prediction == 2:
            result = 'Positive'
        elif prediction == 1:
            result = 'Neutral'
        else:
            result = 'Negative'
            
        return render_template('index.html', prediction=result, review=review_text)

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

