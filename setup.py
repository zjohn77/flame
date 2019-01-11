from setuptools import setup, find_packages

with open("README.md") as f:
   long_description = f.read()

setup(
   name="inflame",
   version="0.1.3",
   license='MIT',
   description="Deep learning applied to text classification, in PyTorch",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author="John Jung",
   author_email="tojohnjung@gmail.com",
   url="https://github.com/zjohn77/inflame",
   packages=find_packages(),
   classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
   ],
   keywords='deep learning'
)