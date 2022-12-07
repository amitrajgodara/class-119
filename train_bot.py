#Text Data Preprocessing Lib
import nltk 
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
import json 
import pickle
import numpy as mp
words=[]
classs=[]
wordtaglist=[]
ignorwards=('?','!',',','.')
traindatafile=open("intents.json").read
intents=json.loads(traindatafile)
# function for appending stem words
def getstemwards(wards,ignorwards):
        stemwards=[]
        for word in words:
                if word not in ignorwards:
                        w=stemmer.stem(word.lower())
                        stemwards.append(w)

        return stemwards
for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            patternword = nltk.word_tokenize(pattern)            
            words.extend(patternword)                      
            wordtaglist.append((patternword, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classs:
            classs.append(intent['tag'])
            stemwords = getstemwards(words, ignorwards)

print(stemwords)
print(wordtaglist[0]) 
print(classs)   

def create_bot_corpus(stemwords, classs):

    stemwords = sorted(list(set(stemwords)))
    classs = sorted(list(set(classs)))

    pickle.dump(stemwords, open('words.pkl','wb'))
    pickle.dump(classs, open('classes.pkl','wb'))

    return stemwords, classs

stemwords, classs = create_bot_corpus(stemwords,classs)  

print(stemwords)
print(classs)

training_data = []
number_of_tags = len(classs)
labels = [0]*number_of_tags

# Create bag od words and labels_encoding
for wordtag in wordtaglist:
        
        bag_of_words = []       
        pattern_words = wordtag[0]
       
        for word in pattern_words:
            index=pattern_words.index(word)
            word=stemmer.stem(word.lower())
            pattern_words[index]=word  

        for word in stemwords:
            if word in pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        print(bag_of_words)

        labels_encoding = list(labels) #labels all zeroes initially
        tag = wordtag[1] #save tag
        tag_index = classs.index(tag)  #go to index of tag
        labels_encoding[tag_index] = 1  #append 1 at that index
       
        training_data.append([bag_of_words, labels_encoding])

print(training_data[0])

# Create training data
def preprocess_train_data(training_data):
   
    training_data = np.array(training_data, dtype=object)
    
    train_x = list(training_data[:,0])
    train_y = list(training_data[:,1])

    print(train_x[0])
    print(train_y[0])
  
    return train_x, train_y

train_x, train_y = preprocess_train_data(training_data)


    
       
