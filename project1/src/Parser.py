#http://tartarus.org/~martin/PorterStemmer/python.txt
from src.PorterStemmer import PorterStemmer
from nltk.tokenize import RegexpTokenizer 


class Parser:

	#A processor for removing the commoner morphological and inflexional endings from words in English
	stemmer=None

	stopwords=[]

	def __init__(self,):
		self.stemmer = PorterStemmer()

		#English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
		self.stopwords = open('./src/english.stop', 'r').read().split()


	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		string = string.replace(".","")
		string = string.replace(r"\s+"," ")
		string = string.lower()
		return string
	

	def removeStopWords(self,list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]


	def tokenise(self, string, stemming=True):
		""" break string up into tokens and stem words """
		string = self.clean(string)
		tk = RegexpTokenizer('\s+', gaps = True) 
		words = tk.tokenize(string) 
  		# words = string.split(" ")
		
		if stemming:
			return [self.stemmer.stem(word,0,len(word)-1) for word in words]
		return words

import jieba
import jieba.posseg as pseg
import re

class JiebaParser:
    stopwords = []

    def __init__(self):
        # Load Chinese stopwords from a file
        # Assuming you have a file named 'chinese.stop' with Chinese stopwords
        self.stopwords = open('./src/english.stop', 'r').read().split()
        # Initialize jieba
        jieba.initialize()

	# Remove Chinese periods, commas, and whitespace
    def clean(self, string: str):
        string = re.sub(r"[。，]", "", string)  # Remove Chinese period and comma
        string = re.sub(r"\s+", " ", string)    # Replace multiple whitespace with a single space
        return string

    def removeStopWords(self, word_list):
        """ Remove common words which have no search value """
        return [word for word in word_list if word not in self.stopwords]
    
    def tokenise(self, text):
        """Tokenize the cleaned text into individual words."""
        cleaned_text = self.clean(text)
        return list(jieba.cut(cleaned_text))

    def tokenise_with_pos(self, text):
        """Tokenize text with part-of-speech tagging."""
        cleaned_text = self.clean(text)
        return list(pseg.cut(cleaned_text))

    def get_nouns_and_verbs(self, text):
        """Extract nouns and verbs from text."""
        pos_tagged_words = self.tokenise_with_pos(text)
        nouns = [word for word, flag in pos_tagged_words if flag.startswith('n')]
        verbs = [word for word, flag in pos_tagged_words if flag.startswith('v')]
        return nouns, verbs