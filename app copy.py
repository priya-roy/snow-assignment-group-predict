import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template, url_for, redirect


#st.set_page_config(page_title='SNOWNLP', page_icon=None, layout='centered', initial_sidebar_state='auto')

# load the model

app = Flask(__name__)


# load the models from disk
model = pickle.load(open('model.pkl', 'rb'))
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
	    data = [message]
	    vect = cv.transform(data).toarray()
	    my_prediction = predict(message)
    
    #print(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == "__main__":
    app.run(debug=True)
