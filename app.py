import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template
import pickle

def lemmatize_text(sentence):
    corpus=[]
    lemmatizer = WordNetLemmatizer() 
    review = re.sub('[^a-zA-Z]', ' ', str(sentence))
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    return corpus
def transform_text(sentence):
    corp = lemmatize_text(sentence)
    vectorizer = pickle.load(open("cv.pickle", 'rb')) 
    X = vectorizer.transform(corp).toarray()
    
    return X
    


app = Flask(__name__)
model = pickle.load(open('nlp_review_sentiment_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    review = request.form.get("reviewtext")
    vector = transform_text(review)
    prediction = model.predict(vector)
    print('review is ----->', prediction)
    if(prediction[0]==5):
         output = "positive"
    elif(prediction[0]==0):
         output = "negative"
    elif(prediction[0]==3):
         output = "neutral"
    elif(prediction[0]==-1):
         output = "irrelavant"
         
    

   

    return render_template('index.html', prediction_text='This review is  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)