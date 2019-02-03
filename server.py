import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for, request, redirect
import pickle
from scipy.sparse import hstack
from link_extract import get_body
from sp_recog import get_speech
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

classifier = {}
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

word_vectorizer = pickle.load(open('pickled_models/word_vectorizer.pkl','rb'))
for class_name in class_names:
    filename = 'pickled_models/model_' + class_name  + '.pkl'
    classifier[class_name] = pickle.load(open(filename,'rb'))

result = {}
for key in class_names:
    result[key] = [False, 0]
default_message = {}

@app.route('/')
def home():
    return render_template('home.html', result=result, default_message=default_message)

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

    classes = {}
    for key in submission.keys():
        classes[key] = [False, 0]
        classes[key][1] = round(submission[key][0] * 100, 0)
        if submission[key][0] > 0.4:
            classes[key][0] = True

    global result, default_message
    result=classes       
    default_message={'api': data, 'link': ''}
    return redirect('/')

@app.route('/link', methods=['POST'])
def predict_from_link():
    # Get the data from the POST request.
    data = get_body(request.form['message'])

    test_pd = pd.DataFrame()
    test_pd['comment_text'] = [data]
    test_pd = test_pd['comment_text']
    
    test_feat = word_vectorizer.transform(test_pd)
    test_feat = hstack([test_feat])

    submission = {}
    for class_name in class_names:                                          
        submission[class_name] = classifier[class_name].predict_proba(test_feat)[:, 1]

    classes = {}
    for key in submission.keys():
        classes[key] = [False, 0]
        classes[key][1] = round(submission[key][0] * 100, 0)
        if submission[key][0] > 0.4:
            classes[key][0] = True
    
    global result, default_message
    result=classes
    default_message={'api': '', 'link': request.form['message']}
    return redirect('/')

@app.route('/upload', methods=['POST'])
def predict_from_upload():
    UPLOAD_FOLDER = './uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    

    data = get_speech(filename)

    test_pd = pd.DataFrame()
    test_pd['comment_text'] = [data]
    test_pd = test_pd['comment_text']
    
    test_feat = word_vectorizer.transform(test_pd)
    test_feat = hstack([test_feat])

    submission = {}
    for class_name in class_names:                                          
        submission[class_name] = classifier[class_name].predict_proba(test_feat)[:, 1]
    
    classes = {}
    for key in submission.keys():
        classes[key] = [False, 0]
        classes[key][1] = round(submission[key][0] * 100, 0)
        if submission[key][0] > 0.4:
            classes[key][0] = True
    
    global result, default_message
    result = classes
    default_message = {'api': '', 'link': ''}
    return redirect('/')
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)