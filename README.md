## 1. The problem this project is trying to solve
1. Can you easily repurpose the deep learning model you just built to new problems?
2. Is the process of converting a deep learning model from development to production a pain point for you?
3. How maintainable is your deep learning model code base?

## 2. The design
The idea of the inflame project (built on top of PyTorch) is to extend PyTorch by allowing one to easily create templates of the deep learning models. Most deep learning model codes are basically prototypes. Having evolved from multiple iterations of trial and error, deep learning model codebases tend to have a lot of hardcoding of values, and they tend to be large single-file beasts. The result is that they are hard to understand, debug, and reuse on new problems. Inflame helps solve this problem by organizing the deep learning model code around concerns. The data, model architecture, hyperparameter, backpropagation, and accuracy evaluation -- each of these domains of functionality is its own module. By organizing and API-ing the code via design patterns, the same deep learning model can be reused from project to project with the smallest number of obvious customizations.

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
