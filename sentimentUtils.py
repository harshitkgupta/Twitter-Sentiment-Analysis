import re
from commonUtils import getSentimentDictionary,getStopWordList,processTweet,replaceTwoOrMore

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    processed_tweet = processTweet(tweet);
    #split tweet into words
    words = processed_tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        stopWordsList = set(getStopWordList())
        if(w in stopWordsList or val is None):
            continue
        else:
            featureVector.append(w)
    return featureVector
#end



def getSentiment(feature_vector):
    pos_neg_count = 0;
    sentiment_dictionary = getSentimentDictionary()
    for word in feature_vector:
        pos_neg_count += sentiment_dictionary.get(word,0); 
    sentiment = 'neutral';    
    if(pos_neg_count > 0):
        sentiment = 'positive'
    elif(pos_neg_count < 0):
        sentiment = 'negative'
    return sentiment 

#start extract_features
def extract_word_features(tweet):
    return dict([(word, True) for word in tweet])
#end

