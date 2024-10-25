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


