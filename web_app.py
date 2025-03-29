from flask import Flask, request, render_template

import pickle
import logging

# Load the trained model and vectorizer
MODEL_PATH = 'model/rf_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'
METRICS_PATH = 'model/metrics.pkl'

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        submission_type = request.form.get('submission_type')

        text = ''
        if submission_type == 'text':
            text = request.form['text']
            logging.info(f"User submitted text: {text}")
            if not text:
                logging.warning("User submitted empty text.")
                return render_template('index.html', error='Please enter or upload text.')
        elif submission_type == 'file':
            if 'file' not in request.files:
                logging.warning("No file part in the request.")
                return render_template('index.html', error='No file selected.')
            file = request.files['file']
            if file.filename == '':
                logging.warning("No file selected for upload.")
                return render_template('index.html', error='No file selected.')
            if file and file.filename.endswith('.txt'):
                try:
                    text = file.read().decode('utf-8')
                    # Log first 100 chars
                    logging.info(
                        f"User uploaded file: {file.filename}, content: {text[:100]}...")
                except Exception as e:
                    logging.error(f"Error reading uploaded file: {e}")
                    return render_template('index.html', error='Error reading the uploaded file.')
            else:
                logging.warning(
                    "Invalid file type. Only .txt files are allowed.")
                return render_template('index.html', error='Invalid file type. Only .txt files are allowed.')
        else:
            logging.warning("Invalid submission type.")
            return render_template('index.html', error='Invalid submission.')

        if text:
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)[0]
            logging.info(f"Prediction: {prediction}")

            if prediction == 'ai':
                result = 'This text is likely AI-generated.'
            else:
                result = 'This text is likely human-written.'

            return render_template('index.html', prediction_result=result, analyzed_text=text)

        return render_template('index.html')


@app.route('/metrics')
def metrics():
    metrics_data = load_metrics()
    if metrics_data:
        return render_template('metrics.html', metrics=metrics_data)
    else:
        return render_template('metrics.html', error='Could not load evaluation metrics.')


def load_metrics():
    try:
        with open(METRICS_PATH, 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        logging.error(f"Error loading metrics: {e}")
        return None


@app.route('/about')
def about():
    return render_template('about.html',
                           github_saif='https://github.com/MohammedSaifAkkiwat',
                           linkedin_saif='https://www.linkedin.com/in/mohammed-saif-akkiwat',
                           github_gokul='https://github.com/GokulKrishnanNair',
                           linkedin_gokul='https://www.linkedin.com/in/gokul-krishnan-nair',
                           github_shreya='https://github.com/ShreyaSinha1309',
                           linkedin_shreya='https://www.linkedin.com/in/shreya-sinha-a05982358')


@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        text = request.form['text']
        prediction = request.form['prediction']
        feedback = request.form['feedback']
        logging.info(
            f"Feedback received: Text='{text[:50]}...', Prediction='{prediction}', Feedback='{feedback}'")
        return render_template('index.html', feedback_submitted=True, analyzed_text=text)
    return render_template('index.html')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    app.run(debug=True)
