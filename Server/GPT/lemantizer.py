 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords

# REFERENCE:  https://barcelonageeks.com/python-enfoques-de-lematizacion-con-ejemplos/
def lemantize(word:str,removeStopWords:bool=True) -> str|list[str]:
    ''' 
    return the lemantization of a sentence
    word: a sentence like 'hello world'
    returnList: a flag for return a list 
    returnPairList: a flag for retun pair list ('flying to florida' -> [('flying','florida)])
    removeStopWords: remover palabras como in out and or etc al utilizar esta opcion se removeran tambien
    los signos de puntuacion para evitar mucho procesamiento
    '''
    lemmatizer = WordNetLemmatizer()
    # Define function to lemmatize each word with its POS tag
    # POS_TAGGER_FUNCTION : TYPE 1
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None

    # tokenize the sentence and find the POS tag for each token
    if removeStopWords:
        # https://randlow.github.io/posts/nlp/bow-nltk/
        word = [ w for w in nltk.word_tokenize(word.lower()) if w.isalpha()]
        word = ' '.join([t for t in word if t not in stopwords.words('english')])
        
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(word.lower()))
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence


def lemantizePairedList(word,removeStopWords:bool=True):
    lemmatized_sentence = lemantize(word,removeStopWords=removeStopWords)
    cindex =0
    indx = [ [sen_a,sen_a+1] if(sen_a+1<=(len(lemmatized_sentence) -1)) else [sen_a,''] for sen_a in range(0,len(lemmatized_sentence),2) ]
    #print(lemmatized_sentence)
    for sen_a,sen_b in indx:
        #print(sen_a,sen_b)
        if not(isinstance(sen_b,str)):# si no es el final
            yield lemmatized_sentence[sen_a],lemmatized_sentence[sen_b]
        else:
            yield lemmatized_sentence[sen_a],sen_b
def lemantizePairedStr(word:str,removeStopWords:bool=True):
    for x,y in lemantizePairedList(word,returnPairList=True,returnList=False,removeStopWords=removeStopWords):
        yield ' '.join([x,y])
def lemantizeStr(word:str,removeStopWords:bool=True):
    return ' '.join(lemantize(word,removeStopWords=removeStopWords))
    

if __name__ == '__main__':
    while True:
        inp = input("#>")
        if inp == 'exit':
            exit(0)
        print(lemantizeStr(inp))
        
        #print([x for x in lemantizePairedStr(inp)])
        #for x,y in lemantize(inp,returnPairList=True,returnList=False):
        #    print(x,y)
        