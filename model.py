import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('./dataset/train.csv').fillna(' ')
test = pd.read_csv('./dataset/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

# Vectorization
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

train_features = hstack([train_word_features])
test_features = hstack([test_word_features])

# Model
scores = []
classifier = {}
for class_name in class_names:
    classifier[class_name] = LogisticRegression(C=0.1, solver='sag')

submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier[class_name].fit(train_features, train_target)
    
    cv_score = np.mean(cross_val_score(classifier[class_name], train_features, train_target, cv=5, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))
    

for class_name in class_names:
    submission[class_name] = classifier[class_name].predict_proba(test_features)[:, 1]
    

print('Total CV score is {}'.format(np.mean(scores)))
submission.to_csv('submission.csv', index=False)

#Pickling
import pickle

pickle.dump(word_vectorizer, open('word_vectorizer.pkl','wb'))
for class_name in class_names:
    filename='model_'+class_name+'.pkl'
    pickle.dump(classifier[class_name], open(filename,'wb'))

# Loading the model
sub={}
for class_name in class_names:
    filename='model_'+class_name+'.pkl'
    f = open(filename, 'rb')
    model=pickle.load(f)
    sub[class_name] = model.predict_proba(t_features)[:, 1]
    
sub