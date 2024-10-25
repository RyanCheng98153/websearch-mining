from pprint import pprint
from src.Parser import Parser
import src.util as util
import math
from typing import Callable
from tqdm import tqdm
import numpy as np
    
class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Tidies terms
    # parser=None

    def __init__(self, documents: dict = {}, use_tqdm=True, parser=None):
        self.use_tqdm = use_tqdm
        self.tfVectors=[]
        self.tfidfVectors = []
        
        self.parser = Parser()
        if parser != None:
            self.parser = parser
        self.doc_keys = list(documents.keys())
        self.doc_values = [documents[k] for k in self.doc_keys]  
        # Extract the document contents (ignoring IDs for now)
        # Ensure that the query terms are also included in the vector space
        if(len(self.doc_values)>0):
            self.build(self.doc_values)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        if self.use_tqdm: print("processing makeTFVector...")
        self.tfVectors = [self.makeTFVector(document) for document in tqdm(documents, disable=not self.use_tqdm)] 
        self.idfVector = self.makeIDFVector(documents)  # Compute IDF vector
        self.tfidfVectors = self.makeTFIDFVectors()  # Compute TF-IDF vectors
        
    def build_query(self, query: str, query_support: str = None):
        if type(query) == str:
            query = self.parser.tokenise(query)
        if type(query_support) == str:
            query_support = self.parser.tokenise(query)
        
        if query_support == None:
            self.query_tfVector = self.makeTFVector(" ".join(query))
            self.query_tfidfVector = [tf * idf for tf, idf in zip(self.query_tfVector, self.idfVector)]
            
        else:
            query_support = list(set(query_support))
            # print(query_support)
            support_tfVector = self.makeTFVector(" ".join(query_support))
            support_tfidfVector = [tf * idf for tf, idf in zip(support_tfVector, self.idfVector)]
            
            self.query_tfVector = [ q+0.5*sq for q, sq in zip(self.query_tfVector, support_tfVector)]
            self.query_tfidfVector = [ q+0.5*sq for q, sq in zip(self.query_tfidfVector, support_tfidfVector)]

        
    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)
        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        if self.use_tqdm: print("processing uniqueVocabularyList...")
        for word in tqdm(uniqueVocabularyList, disable=not self.use_tqdm):
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)

    def makeTFVector(self, wordString, normalized=False):
        """ Create the raw TF (term frequency) vector """
        tfVector = [0] * len(self.vectorKeywordIndex)  # Initialize with 0's
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        
        for word in wordList:
            # if word in self.vectorKeywordIndex.keys():
            if word in self.vectorKeywordIndex.keys():
                tfVector[self.vectorKeywordIndex[word]] += 1  # Count occurrences
                
        # Normalize if required
        if normalized:
            total_terms = sum(tfVector)
            if total_terms > 0:
                tfVector = [count / total_terms for count in tfVector]
        
        return tfVector
    
    def makeIDFVector(self, documents):
        """ Create the IDF (inverse document frequency) vector """
        if self.use_tqdm: print("processing makeIDFVector...")
        idfVector = [0] * len(self.vectorKeywordIndex)
        # print(self.vectorKeywordIndex)
        for doc in tqdm(documents, disable=not self.use_tqdm):
            words = self.parser.tokenise(doc)
            words = self.parser.removeStopWords(words)
            for word in set(words):
                if word in self.vectorKeywordIndex.keys():
                    idfVector[self.vectorKeywordIndex[word]] += 1
                
        docN = len(documents)
        
        idfVector = [math.log(docN / contain) if contain != 0 else 0 for contain in idfVector]
        # euclidean_idfVector = [math.log(docN / 1 + contain) for contain in idfVector]
        
        return idfVector
    
    def makeTFIDFVectors(self):
        """ Create the TF-IDF vectors for each document """
        tfidfVectors = []
        if self.use_tqdm: print("processing makeTFIDFVectors...")
        
        for tfVector in tqdm(self.tfVectors, disable=not self.use_tqdm):
            tfidfVector = [tf * idf for tf, idf in zip(tfVector, self.idfVector)]
            tfidfVectors.append(tfidfVector)
        
        return tfidfVectors
    
    def search(self, query:str="", query_support:str=None, method="cosine", use_tfidf=False, topN_results=-1):
        """ Search for documents that match based on the chosen similarity method and vector type (TF or TF-IDF) """
        self.build_query(query=query, query_support=query_support)
        
        if use_tfidf:
            documentVectors = self.tfidfVectors
            queryVector = self.query_tfidfVector    
        else:
            documentVectors = self.tfVectors
            queryVector = self.query_tfVector
        
        if self.use_tqdm: print(f"processing {'TF-IDF' if use_tfidf else 'TF'} {method} ratings..")
        if method == "cosine":
            ratings = {dkey: self.cosine_similarity(queryVector, dVec) for dkey, dVec in tqdm(zip(self.doc_keys, documentVectors), total=len(self.doc_keys), disable=not self.use_tqdm)}
        elif method == "euclidean":
            ratings = {dkey: self.euclidean_distance(queryVector, dVec) for dkey, dVec in tqdm(zip(self.doc_keys, documentVectors), total=len(self.doc_keys), disable=not self.use_tqdm)}
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cosine' or 'euclidean'.")

        # Sort the ratings by value (similarity score) in descending order
        sorted_ratings = sorted(ratings.items(), key=lambda item: item[1], reverse=(method == "cosine") )

        if topN_results == -1:
            return sorted_ratings
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
        
        if a.size == 0 or np.linalg.norm(a) == 0.0:
            return 0.0
        
        if b.size == 0 or np.linalg.norm(b) == 0.0:
            return 0.0
        # Calculate cosine similarity
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def euclidean_distance(self, vector1, vector2):
        """ Calculate Euclidean Distance between two vectors """
        return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))
    
    def mrr_at_k(self, relevance_scores, k=10):
        for i, score in enumerate(relevance_scores[:k]):
            if score == 1:  # Assuming 1 for relevant, 0 for not relevant
                return 1 / (i + 1)
        return 0.0
        
    def average_precision_at_k(self, relevance_scores, k=10):
        num_relevant = 0
        score_sum = 0.0
        for i, score in enumerate(relevance_scores[:k]):
            if score == 1:
                num_relevant += 1
                score_sum += num_relevant / (i + 1)
        return score_sum / min(sum(relevance_scores), k)
    
    def recall_at_k(self, relevance_scores, k=10):
        relevant_retrieved = sum(relevance_scores[:k])
        total_relevant = sum(relevance_scores)
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0


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
