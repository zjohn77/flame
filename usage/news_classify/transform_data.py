def reshape(data):
   '''Transform data (in dict form) into 2 lists for supervised learning.
   The lists are: target (type of news) & features (news content),
   represented by y & x respectively. Elements in y can be {0, 1, 2,...N}, where N
   is the number of news categories.
   '''
   x = []
   y = []
   for i, value in enumerate(data.values()):
      x += value
      y += [i]*len(value)
   return x, y