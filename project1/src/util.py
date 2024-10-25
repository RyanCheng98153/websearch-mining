import sys
import os

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))

def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def read_files(dirpath):
    documents = dict()
    for file in os.listdir(dirpath):
        # print(file)
        if not file.endswith(".txt"):
            continue
        with open(dirpath + file, 'r', encoding="utf-8") as f: 
            documents[file] = "".join(f.readlines())
    return documents

def read_file(filepath):
    # print(file)
    with open(filepath, 'r', encoding="utf-8") as f: 
        document = "".join(f.readlines())
    return document

from nltk import word_tokenize, pos_tag

def getVerbNoun(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # POS tagging
    tagged_tokens = pos_tag(tokens)

    # Extract nouns and verbs
    nouns = [word for word, pos in tagged_tokens if pos.startswith('NN')]
    verbs = [word for word, pos in tagged_tokens if pos.startswith('VB')]
    
    return nouns + verbs


from typing import Callable

# import tqdm
def tqdm(func: Callable, total=False): 
    '''
    : a spooky method to pass the tqdm function
    : but return the parameter list or function inside
    '''
    # print(type(func))
    # if type(func) is list or type(func) is set or type(func) is type({}.keys()) or type(func) is zip :
    if type(func) is not Callable:
        return func
    def inner(*args, **kwargs):
        return func(*args, **kwargs)
    return inner
