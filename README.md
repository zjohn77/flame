[![License](https://img.shields.io/github/license/zjohn77/inflame.svg)](https://github.com/zjohn77/inflame/blob/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/inflame.svg)](https://pypi.org/project/inflame/)


## 1. The problem this project is trying to solve
1. Can you easily repurpose the deep learning model you just built to a new problem?
2. Is converting a deep learning model from development to production a cumbersome task for you?
3. How maintainable is your deep learning model code base?

## 2. The design
The inflame project is intended to extend PyTorch by facilitating easy templating of deep learning models. Most deep learning model codes tend to have a lot of hardcoding of values, and they tend to be large single-file beasts, with the result that they are hard to understand, debug, and reuse on new problems. Inflame helps solve this problem by organizing the deep learning model code around concerns. The data, model architecture, hyperparameter, backpropagation, and accuracy evaluation -- each of these domains of functionality is its own module. By organizing and API-ing the code via design patterns, the same deep learning model can be reused from project to project with the smallest number of customizations.

## 3. Install
```sh
pip install inflame
```

## 4. Quick start
To run the demo project that classifies news articles using convolutional net:
```sh
inflame_run --corpus bbcnews
```
To run the demo project that classifies newsgroup posts using convolutional net:
```sh
inflame_run --corpus newsgrp
```
Read [this guide](https://zjohn77.github.io/blog/posts/convnet/) to understand the design, and for an example of using the inflame package for text analysis.

## 5. Contribute
Any contribution is welcome. To get started:
```sh
git clone https://github.com/zjohn77/inflame.git
pip install -r requirements.txt
cd test && python -m unittest
```