import re
import os
import cPickle


#put Name of all Serialized objects
SentimentDictPkl = os.path.join('pkl','SentimentDict.pkl')
PositiveFeaturesPkl = os.path.join('pkl','PositiveFeatures.pkl')
NegativeFeaturesPkl = os.path.join('pkl','NegativeFeatures.pkl')
StopWordsPkl = os.path.join('pkl','StopWords.pkl')
FeaturesListPkl = os.path.join('pkl','FeaturesList.pkl')

lexiconFileName = 'lexicon.txt'
stopWordsFileName = 'stopwords.txt'

#start getStopWordList
def getStopWordList():
    if not os.path.exists('./%s'%StopWordsPkl):       
        #read the stopwords file and build a list
        stopWords = []
        stopWords.append('AT_USER')
        stopWords.append('URL')
    
        fp = open(stopWordsFileName, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
        cPickle.dump(stopWords,file(StopWordsPkl, 'wb'))
    else:
        stopWords = cPickle.load(file(StopWordsPkl,'rb'))
    return stopWords


#This method will return sentiment dictionary of postive and negative words
def getSentimentDictionary():
    if not os.path.exists('./%s'%SentimentDictPkl):
        sentiment_dictionary = {}
        lexicon = open(lexiconFileName,'r');
        for line in  lexicon.read().split('\r'):
            if line.endswith('positive'):
                positive_word = line.split('\t')[0]; 
                sentiment_dictionary[positive_word] = 1;
            elif line.endswith('negative'):
                negative_word = line.split('\t')[0]; 
                sentiment_dictionary[negative_word] = -1;
        cPickle.dump(sentiment_dictionary,file(SentimentDictPkl, 'wb'))
    else:
        sentiment_dictionary = cPickle.load(file(SentimentDictPkl,'rb'))
        
    return sentiment_dictionary 

def getPositiveFeatures():
    if not os.path.exists('./%s'%PositiveFeaturesPkl):
        positive_features = []
        lexicon = open(lexiconFileName,'r');
        for line in  lexicon.read().split('\r'):
            if line.endswith('positive'):
                positive_word = line.split('\t')[0]; 
                positive_features.append(positive_word);               
        cPickle.dump(positive_features,file(PositiveFeaturesPkl, 'wb'))
    else:
        positive_features = cPickle.load(file(PositiveFeaturesPkl,'rb'))
        
    return positive_features        

def getNegativeFeatures():
    if not os.path.exists('./%s'%NegativeFeaturesPkl):
        negative_features = []
        lexicon = open(lexiconFileName,'r');
        for line in  lexicon.read().split('\r'):
            if line.endswith('negative'):
                negative_word = line.split('\t')[0]; 
                negative_features.append(negative_word);               
        cPickle.dump(negative_features,file(NegativeFeaturesPkl, 'wb'))
    else:
        negative_features = cPickle.load(file(NegativeFeaturesPkl,'rb'))
        
    return negative_features 


def getAllFeaturesList():
    if not os.path.exists('./%s'%FeaturesListPkl):
        allFeaturesList = []  
        allFeaturesList.extend(getPositiveFeatures())
        allFeaturesList.extend(getNegativeFeatures())      
        #remove duplicates from feature List
        allFeaturesList = list(set(allFeaturesList));         
        cPickle.dump(allFeaturesList,file(FeaturesListPkl, 'wb'))
    else:
        allFeaturesList = cPickle.load(file(FeaturesListPkl,'rb'))
        
    return allFeaturesList


# start process tweet
def processTweet(tweet):
    #convert to lower case
    tweet = tweet.lower();
    
    ##Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end
   
