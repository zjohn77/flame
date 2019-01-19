## 1. The problem this project is trying to solve
1. Can you easily repurpose the deep learning model you just built to new problems?
2. Is the process of converting a deep learning model from development to production a pain point for you?
3. How maintainable is your deep learning model code base?

PyTorch is among the best deep learning frameworks out there. It's relatively easy to use and the coding style is readable and pythonic. Nevertheless, there's still a fair amount of boilerplate code and hard-coded numbers in a typical pytorch project. The goal of this project (a.k.a. inflame) is to abstract away the pieces that remain the same from model to model, so that the same deep learning template, or archetype, can be quickly reapplied to new problems. 

## 2. Install
```sh
pip install inflame
```

## 3. Quick Start
```sh
python -m inflame news
python -m inflame newsgrp
```