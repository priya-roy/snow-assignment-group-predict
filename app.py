import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report, accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify, render_template, url_for, redirect


#st.set_page_config(page_title='SNOWNLP', page_icon=None, layout='centered', initial_sidebar_state='auto')

# load the model

app = Flask(__name__)


# load the models from disk
model = pickle.load(open('clf.pkl', 'rb'))
cv    = pickle.load(open("cv.pkl",'rb'))
le    = pickle.load(open("le.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# Prediction
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['short_description']
        data = cv.transform([message]).toarray()
        assg = model.predict(data)
        assignment_group = le.inverse_transform(assg)
    return render_template('home.html',prediction = assignment_group[0], message = message)


if __name__ == "__main__":
    app.run(debug=True)
