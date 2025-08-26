from flask import Flask, render_template, request, jsonify
from webapp import model_loader

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    model_choice = request.form.get("model")  # Get selected model

    # Load vectorizer and chosen model
    vec, clf = model_loader.load(model_choice)

    # Transform and predict
    X = vec.transform([text])
    pred = clf.predict(X)[0]

    return jsonify({"prediction": pred})


if __name__ == "__main__":    # <---- THIS IS IMPORTANT
    app.run(host="127.0.0.1", port=5000, debug=True)
