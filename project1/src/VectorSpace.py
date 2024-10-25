from pprint import pprint
from src.Parser import Parser
import src.util as util
import math
from typing import Callable

from tqdm import tqdm
def tqdm2(func: Callable, total=False): 
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

import numpy as np
    
class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Tidies terms
    parser=None

    def __init__(self, documents: dict = {}, query=[], use_tqdm=True):
        self.use_tqdm = use_tqdm
        self.tfVectors=[]
        self.tfidfVectors = []
        self.parser = Parser()
        if type(query) == str:
            query = self.parser.tokenise(query)
        self.doc_keys = list(documents.keys())
        self.doc_values = [documents[k] for k in self.doc_keys]  
        # Extract the document contents (ignoring IDs for now)
        # Ensure that the query terms are also included in the vector space
        if(len(self.doc_values)>0):
            self.build(self.doc_values, query)

    def build(self,documents, query):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents + query)
        if self.use_tqdm: print("processing makeTFVector...")
        self.tfVectors = [self.makeTFVector(document) for document in tqdm(documents)] 
        self.idfVector = self.makeIDFVector(documents)  # Compute IDF vector
        self.tfidfVectors = self.makeTFIDFVectors()  # Compute TF-IDF vectors
        
        self.query_tfVector = self.makeTFVector(" ".join(query))
        self.query_tfidfVector = [tf * idf for tf, idf in zip(self.query_tfVector, self.idfVector)]
        
        # print(self.tfVectors)
        # print(self.idfVector)
        

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)
        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        # vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        if self.use_tqdm: print("processing uniqueVocabularyList...")
        for word in tqdm(uniqueVocabularyList):
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)

    def makeTFVector(self, wordString, normalized=False):
        """ Create the raw TF (term frequency) vector """
        tfVector = [0] * len(self.vectorKeywordIndex)  # Initialize with 0's
        wordList = self.parser.tokenise(wordString)
        
        for word in wordList:
            # if word in self.vectorKeywordIndex.keys():
            tfVector[self.vectorKeywordIndex[word]] += 1  # Count occurrences
                
        # Normalize if required
        if normalized:
            total_terms = sum(tfVector)
            if total_terms > 0:
                tfVector = [count / total_terms for count in tfVector]
        
        return tfVector
    
    def n_containing(self, word, bloblist):
        """ Count how many documents contain the term """
        return sum(1 for blob in bloblist if word in blob)

    def makeIDFVector(self, documents):
        """ Create the IDF (inverse document frequency) vector """
        if self.use_tqdm: print("processing makeIDFVector...")
        idfVector = [0] * len(self.vectorKeywordIndex)
        # print(self.vectorKeywordIndex)
        for word in tqdm(self.vectorKeywordIndex.keys()):
            # print((1 + self.n_containing(word, documents)))
            idfVector[self.vectorKeywordIndex[word]] = math.log(len(documents) / (1 + self.n_containing(word, documents)))
            # Adding 1 to prevent division by zero
        
        return idfVector
    
    def makeTFIDFVectors(self):
        """ Create the TF-IDF vectors for each document """
        tfidfVectors = []
        if self.use_tqdm: print("processing makeTFIDFVectors...")
        # print(self.tfVectors[0])
        # print(self.idfVector[0])
        for tfVector in tqdm(self.tfVectors):
            tfidfVector = [tf * idf for tf, idf in zip(tfVector, self.idfVector)]
            tfidfVectors.append(tfidfVector)
        
        return tfidfVectors
    
    def search(self, method="cosine", use_tfidf=False, topN_results=10):
        """ Search for documents that match based on the chosen similarity method and vector type (TF or TF-IDF) """
        queryVector = self.query_tfidfVector if use_tfidf else self.query_tfVector
        documentVectors = self.tfidfVectors if use_tfidf else self.tfVectors
        
        if self.use_tqdm: print(f"processing {'TF-IDF' if use_tfidf else 'TF'} {method} ratings..")
        if method == "cosine":
            ratings = {dkey: self.cosine_similarity(queryVector, documentVector) for dkey, documentVector in tqdm(zip(self.doc_keys, documentVectors), total=len(self.doc_keys))}
        elif method == "euclidean":
            ratings = {dkey: self.euclidean_distance(queryVector, documentVector) for dkey, documentVector in tqdm(zip(self.doc_keys, documentVectors), total=len(self.doc_keys))}
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cosine' or 'euclidean'.")

        # Sort the ratings by value (similarity score) in descending order
        sorted_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=(method == "cosine") )

        return sorted_ratings[:topN_results]

    def cosine_similarity_0(self, vector1, vector2):
        """ Calculate Cosine Similarity between two vectors """
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(v ** 2 for v in vector1))
        magnitude2 = math.sqrt(sum(v ** 2 for v in vector2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
    
    def cosine_similarity(self, vector1, vector2):
        """ Calculate Cosine Similarity between two vectors """
        a = np.array(vector1)
        b = np.array(vector2)
        # Calculate dot product
        dot_product = np.dot(a, b)

        # Calculate magnitudes
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)

        # Handle the case where the magnitude is zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0  # or you can raise an exception

        # Calculate cosine similarity
        cosine_sim = dot_product / (magnitude_a * magnitude_b)

        return cosine_sim
        

    def euclidean_distance(self, vector1, vector2):
        """ Calculate Euclidean Distance between two vectors """
        return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))

    

if __name__ == '__main__':
    #test data
    documents = ["The cat in the hat disabled",
                 "A cat is a fine pet ponies.",
                 "Dogs and cats make good pets.",
                 "I haven't got a hat."]

    vectorSpace = VectorSpace(documents)

    #print(vectorSpace.vectorKeywordIndex)

    #print(vectorSpace.documentVectors)

    print(vectorSpace.related(1))

    #print(vectorSpace.search(["cat"]))

###################################################
