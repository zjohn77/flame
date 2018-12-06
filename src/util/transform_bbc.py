def reshape(data):
   x = []
   y = []
   for i, value in enumerate(data.values()):
      x += value
      y += [i]*len(value)
   return x, y   