var Tokenizer = function() {
};

Tokenizer.prototype.trim = function(array) {
  while (array[array.length - 1] == '')
    array.pop();

  while (array[0] == '')
    array.shift();

  return array;
};

// Expose an attach function that will patch String with new methods.
Tokenizer.prototype.attach = function() {
  var self = this;

  String.prototype.tokenize = function() {
    return self.tokenize(this);
  }
};

Tokenizer.prototype.tokenize = function() {};

module.exports = Tokenizer;