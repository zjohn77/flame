"""
Fetch data from 20newsgroups in sklearn.datasets with these key attributes: "data", "target".
Keep only texts with no more than 1000 tokens. Embed into 50D GloVe.
"""
from numpy import zeros
from gensim.utils import tokenize
from gensim.downloader import load
class TextData:
   def __init__(self, data, target):
      self.__WORD_VECS = load("glove-wiki-gigaword-50") ## Set an embedder that takes a text arg & embeds it.
      self.__MAXLEN = 200
      self.__data = data
      self.__target = target
      self.__embedding = zeros((len(data), self.__MAXLEN, self.__WORD_VECS.vector_size))
      # The self.__embedding prefills each tensor with 0's to ensure that all are the
      # same size regardless of text lengths. 
      
   def gen_tokens(self, text):
      '''tokenize and then filter out stopwords'''
      return filter(lambda token: token not in {'a', 'and', 'for', 'in', 'of', 'the', 'to'}, 
                    tokenize(text, lower=True)
                   )

   def embed(self):   
      '''Chunk each document into words and then embed those words into 300D GloVe vectors.
      Stack the embedded numeric document matrices into a numpy tensor (i.e. our predictors).
      '''        
      for i, doc in enumerate(self.__data):
         words = [word for word in self.gen_tokens(doc) if word in self.__WORD_VECS.vocab]
         J = min(len(words), self.__MAXLEN)
         for j in range(J):
            # print(type(words[j]))
            self.__embedding[i, j] = self.__WORD_VECS[words[j]]

   def getter(self):
      return self.__data, self.__target, self.__embedding