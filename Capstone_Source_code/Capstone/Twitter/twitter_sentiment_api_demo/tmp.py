from flask import Flask, render_template, request, redirect, url_for
import jinja2
from joblib import load
from get_tweet_data import get_data
import json
import tweepy 
from tweepy.auth import OAuthHandler
app = Flask(__name__,template_folder="template")

@app.route('/')
def base():
    # ENV = jinja2.Environment(loader=jinja2.FileSystemLoader('twitter_sentiment_api_demo/template'))
    # template = ENV.get_template('base.html')

    return render_template("tweets.html")

@app.route('/',methods=['POST','GET'])
def get_tweets():
    success=False
    if request.method == 'POST':
        topic = request.form['search']
        print(type(topic))
        try:
            global tweets
            tweets = get_data(topic)
            print(tweets)
            if len(tweets)>0:
                success = True
                return render_template('tweets.html',status = 'successfull',show=False)
            else:
                return render_template('tweets.html',status = 'unsuccessfull')
        except:
            return render_template('tweets.html',status = 'something went wrong')

@app.route('/show',methods = ["POST","GET"])
def show():
    print(tweets[0])
    if request.method=='GET':
        return render_template('tweets.html',status = 'successfull',show=True,tweets=tweets)

@app.route('/pipeline')
def pipeline():
    import datetime
    print(tweets)
    def myconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()
 
    return render_template('pipeline.html',tweets=json.dumps(tweets, default = myconverter))

if __name__ == "__main__":
    app.run(debug=True)