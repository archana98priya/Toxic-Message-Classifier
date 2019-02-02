import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for, request
import pickle
from scipy.sparse import hstack

app = Flask(__name__)

classifier = {}
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

word_vectorizer = pickle.load(open('pickled_models/word_vectorizer.pkl','rb'))
for class_name in class_names:
    filename = 'pickled_models/model_' + class_name  + '.pkl'
    classifier[class_name] = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.form['message']

    test_pd = pd.DataFrame()
    test_pd['comment_text'] = [data]
    test_pd = test_pd['comment_text']
    
    test_feat = word_vectorizer.transform(test_pd)
    test_feat = hstack([test_feat])

    submission = {}
    for class_name in class_names:                                          
        submission[class_name] = classifier[class_name].predict_proba(test_feat)[:, 1]

    for key in submission.keys():
        submission[key] = submission[key][0]
   
    return jsonify(submission)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)