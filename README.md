## 1. The Goal
PyTorch is among the best deep learning frameworks out there. It's relatively easy to use and the coding style is readable and pythonic. Nevertheless, there's still a fair amount of boilerplate code and hard-coded numbers in a typical pytorch project. The goal of this project (a.k.a. inflame) is to abstract away the pieces that remain the same from model to model, so that the same deep learning template, or archetype, can be quickly reapplied to new problems. 

## 2. Quick Start
```sh
cd usage
python news_classify
python newsgrp_classify

cd test
python -m unittest discover -v
```