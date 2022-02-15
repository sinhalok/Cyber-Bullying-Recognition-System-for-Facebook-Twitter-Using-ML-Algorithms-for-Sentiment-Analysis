import nltk
#nltk.download('sentiwordnet')

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Basic Packages
import random
import pandas as pd

# Text Preprocessing Packages
import re
import nltk
# from nltk.tokenize import word_tokenize
import tweepy 
from tweepy.auth import OAuthHandler
from textblob import *

import warnings
warnings.filterwarnings("ignore")

def get_data(name):
    def initialize(): 

        # keys and tokens from the Twitter Dev Console 
        consumer_key = 'snseusIoIioTvEpDaBcPjUryw'
        consumer_secret = 'cDhGVySW9xRUQSbc2o8yKMHfAxBrnIBvE1wSaWoz1PIBXspFTm'
        access_token = '2856915038-5xVKLBpmMhX3l4uHBZfmdmIRtXGt1Q0K8yUrexR'
        access_token_secret = 'uiHGPQZuxr9z0bsnaPMPgdMemk8oXOwUYjSLJfsT2VmCM'

        # attempt authentication 
        try: 
            # create OAuthHandler object 
            auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            api = tweepy.API(auth) 
            print('Authentication Success')
            return(api)

        except Exception as e: 
            print("Error: Authentication Failed")
            print(e)


    def get_tweets(api, query, count = 100): 
            # empty list to store parsed tweets 
            tweets = [] 
            sinceId = None
            max_id = -1
            tweetCount = 0
            tweetsPerQry = 100

            while tweetCount < count:
                try:
                    if (max_id <= 0):
                        if (not sinceId):
                            new_tweets = api.search(q=query, count=tweetsPerQry,lang='en')
                        else:
                            new_tweets = api.search(q=query, count=tweetsPerQry,
                                                    since_id=sinceId,lang='en')
                    else:
                        if (not sinceId):
                            new_tweets = api.search(q=query, count=tweetsPerQry,
                                                    max_id=str(max_id - 1),lang='en')
                        else:
                            new_tweets = api.search(q=query, count=tweetsPerQry,
                                                    max_id=str(max_id - 1),
                                                    since_id=sinceId,lang='en')
                    if not new_tweets:
                        print("No more tweets found")
                        break

                    for tweet in new_tweets:
                        parsed_tweet = {} 
                        parsed_tweet['tweets'] = tweet.text
                        parsed_tweet['date'] = tweet.created_at

                        # saving sentiment of tweet 
                        #parsed_tweet['sentiment_score'],parsed_tweet['sentiment'] = get_sentiment(tweet.text)
                        #parsed_tweet['sentiments'] = [tag_sentiment(tweet.text)]
                        # appending parsed tweet to tweets list 
                        if tweet.retweet_count > 0: 
                            # if tweet has retweets, ensure that it is appended only once 
                            if parsed_tweet not in tweets: 
                                tweets.append(parsed_tweet) 
                        else: 
                            tweets.append(parsed_tweet) 

                    tweetCount += len(new_tweets)
                    #print("Downloaded {0} tweets".format(tweetCount))
                    max_id = new_tweets[-1].id
                   # print(max_id)
                   # print(new_tweets[-1])
                    return tweets
                except tweepy.TweepError as e:
                    print("Tweepy error : " + str(e))

    api_initialization = initialize()
    retreived_tweets = get_tweets(api=api_initialization,query=name, count = 100)
    
    return retreived_tweets