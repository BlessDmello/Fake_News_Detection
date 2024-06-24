import numpy as np
from flask import Flask, request,render_template
from flask import Flask
from flask_cors import CORS
import os
import joblib
#from sklearn.externals import joblib
import pickle
import flask
import os
from newspaper import Article
import urllib
import nltk
nltk.download('punkt')
#Loading Flask and assigning the model variable
app = Flask (__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')


model = joblib.load('D:/FAKE_NEWS_DETECTION-MAIN/model.pkl')

#with open('model.pkl', 'rb') as handle:
     #model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('index.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    print("news is",news)

   #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))

@app.route('/summarization',methods=['GET', 'POST'])
def summarization():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    aut=article.authors
    pub=article.publish_date
    sum=article.summary

    return render_template('index.html', summary=sum,auther=aut,date=pub)

if __name__ =="__main__":
   port=int(os.environ.get('PORT',5000))
   app.run(port=port,debug=True,use_reloader=False)