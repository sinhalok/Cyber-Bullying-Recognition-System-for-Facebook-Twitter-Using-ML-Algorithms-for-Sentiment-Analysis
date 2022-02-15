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
    #name='gandhi'
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
                        parsed_tweet['cleaned_tweets'],parsed_tweet['sentiment_score'],parsed_tweet['sentiment'] = get_sentiment(tweet.text)
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
    
    def clean_tweet(tweets): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        #print(tweets)
        return(' '.join(re.sub("([,\.():;!$%^&*\d])|([^0-9A-Za-z \t])", " ", tweets).split())) 

    def penn_to_wn(tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    def get_sentiment(text):
        """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """
        sentiment_score = []
        sentiment = []
        cleaned_tweets = preprocess_tweet(text)
        analysis = TextBlob(clean_tweet(cleaned_tweets)) 
    # set sentiment
        if analysis.sentiment.polarity > 0:
            sentiment_score.append(round(analysis.sentiment.polarity,3))
            sentiment.append('positive')
        elif analysis.sentiment.polarity == 0:
            sentiment_score.append(round(analysis.sentiment.polarity,3))
            sentiment.append('neutral')
        else: 
            sentiment_score.append(round(analysis.sentiment.polarity,3))
            sentiment.append('negative')

        return cleaned_tweets,sentiment_score.pop(), sentiment.pop()

    contractions = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

    def expand_contractions(text):
        for word in text.split():
            if word.lower() in contractions:
                text = text.replace(word, contractions[word.lower()])
        return text

    def preprocess_word(word):
        # Remove punctuation
        word = word.strip('"?!,.():;')
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-)', '', word)
        return word

    def is_valid_word(word):
        # Check if word begins with an alphabet
        return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

    def handle_emojis(tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
        return tweet
    from nltk import WordNetLemmatizer

    def preprocess_tweet(tweet):
        processed_tweet = []
        # Convert to lower case
        tweet = tweet.lower()
        tweet = expand_contractions(re.sub('’', "'", tweet))
        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', r' ', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Strip space, " and ' from tweet
        #tweet = tweet.strip(' "\'')
        # Replace emojis with either EMO_POS or EMO_NEG
        tweet = handle_emojis(tweet)
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = re.sub(r'(\[|\])',' ', tweet)
        
        words = tweet.split()

        for word in words:
            word = preprocess_word(word)
            if is_valid_word(word):
                #if use_stemmer:
                word = str(WordNetLemmatizer().lemmatize(word))
                processed_tweet.append(word)

        return ' '.join(processed_tweet)

    global retreived_tweets
    retreived_tweets = get_tweets(api=api_initialization,query=name, count = 100)

        
    return retreived_tweets