import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = [] #List of dictionaries, were each key is a word and the key is the Log Likehood
    guesses = [] #List of words guessed

    for seq in test_set.get_all_sequences(): #Looping on each seq in all sequences
        
        X, length = test_set.get_item_Xlengths(seq)
        guessWord=''
        maximum = float('-inf')
        probabDict = {}
        
        for keyWord in models: #looping on each model
            
            try:
                
                score= models[keyWord].score(X, length) #Calculating the Score
                probabDict[keyWord] = score 
                
            except:
                
                probabDict[keyWord] = float('-inf') 
                
            if score >= maximum: #Setting the word with the maximum value
                
                maximum = score
                guessWord = keyWord  
                
        guesses.append(guessWord)
        probabilities.append(probabDict)

    return (probabilities, guesses)
