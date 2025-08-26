# Cyber Abuse Detection (Roman Urdu)

This project detects cyber abuse in Roman Urdu text using machine learning and deep learning models. It covers data preprocessing, feature extraction, model training, evaluation, and a Flask web app for real-time predictions.

---

## Features

- **Data Preprocessing:** Cleans and prepares Roman Urdu comments.
- **Feature Extraction:** TF-IDF and BERT embeddings.
- **Model Training:** MultinomialNB, LinearSVC, RandomForest, BERT-BiLSTM, and BERT-BiLSTM-Attention.
- **Evaluation:** Confusion matrices and metrics.
- **Web App:** Flask interface for real-time abuse detection.

---

## Project Structure
```
Cyber Abuse Detection Using ML In Roman Urdu/
├── data/
│   └── cyber_abuse_data_roman_urdu.csv
├── models/
│   ├── mhsCad_naive_bayes.pkl
│   ├── mhsCad_svm.pkl
│   ├── mhsCad_random_forest.pkl
│   ├── mhsCad_best_bert.pt
│   └── mhsCad_best_bert_tokenizer/
├── outputs/
├── src/
│   ├── config.py
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── webapp/
│   ├── static/
│   │   ├── style.css
│   │   └── images/
│   │       └── favicon.ico
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   └── model_loader.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Usage

1. **Install dependencies:**
- pip install -r requirements.txt

2. **Prepare your dataset:**
- Place your data at `data/cyber_abuse_data_roman_urdu.csv` with columns: `tweets,label` (labels: `A` for abusive, `NA` for not abusive).

3. **Train models:**
python src/cyber_abuse_detection.py

4. **Run the web app:**
cd src/webapp python app.py
- Open your browser at `http://127.0.0.1:5000/` to use the interface.

---

## Notes

- Ensure all model files are present in the `models/` directory for prediction.
- If you get "nan" as output, check your preprocessing and label mapping logic.

---

## License

MIT