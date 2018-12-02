"""
Fetch data from 20newsgroups in sklearn.datasets with these key attributes: "data", "target".
Keep only texts with no more than 1000 tokens. Embed into 300D GloVe.
"""
from numpy import zeros
from spacy import load

PRETRAINED_GLOVE = load('en_vectors_web_lg') ## Set an embedder that takes a text arg & embeds it.

class TextData:
   def __init__(self, data, target):
      self.__MAXLEN = 28
      self.__data = data
      self.__target = target
      self.__embedding = zeros((len(data), self.__MAXLEN, 300))
   
   def embed(self):   
      '''Chunk each document into words and then embed those words into 300D GloVe vectors.
      Stack the embedded numeric document matrices into a numpy tensor (i.e. our predictors).
      '''        
      for i, doc in enumerate(self.__data):
         words = PRETRAINED_GLOVE(doc)
         J = min(len(words), self.__MAXLEN)
         for j in range(J):
            self.__embedding[i, j] = words[j].vector

   def getter(self):
      return self.__data, self.__target, self.__embedding

# if __name__ == '__main__':
#    textdata = TextData(__NEWSGROUPS.data, __NEWSGROUPS.target)
#    textdata.embed()
#    data, target, embedding = textdata.getter()   