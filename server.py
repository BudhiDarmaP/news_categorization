from flask import Flask, render_template, request
import json
from joblib import load

app = Flask(__name__)

labelEncoder = load("prod/labelEncoder.joblib")
vectorizer = load("prod/tfidfvectorizer-prod.joblib")

svm = load("prod/svm-rbf-prod.joblib")
gnb = load("prod/gnb-prod.joblib")
mnb = load("prod/mnb-prod.joblib")
bnb = load("prod/bnb-prod.joblib")

@app.route("/", methods=['POST', 'GET'])
def home_page():
  if request.method == "GET":
    return render_template("home.html")
  elif request.method == "POST":
    model = request.form['model']
    text = request.form['text']
    X = vectorizer.transform([text])

    pred = None
    if model == "svm":
      pred = svm.predict(X)
    elif model == "gnb":
      pred = gnb.predict(X.toarray())
    elif model == "mnb":
      pred = mnb.predict(X)
    elif model == "bnb":
      pred = bnb.predict(X)
    else:
      return

    return {
      'text': text,
      'model': model,
      'classification': labelEncoder.inverse_transform(pred).tolist()
    }