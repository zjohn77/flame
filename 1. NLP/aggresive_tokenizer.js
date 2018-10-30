var Tokenizer = require('./tokenizer'),
    util = require('util');

var AggressiveTokenizer = function() {
    Tokenizer.call(this);    
};
util.inherits(AggressiveTokenizer, Tokenizer);

module.exports = AggressiveTokenizer;

AggressiveTokenizer.prototype.tokenize = function(text) {
    // break a string up into an array of tokens by anything non-word
    return this.trim(text.split(/\W+/));
};