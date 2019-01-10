from setuptools import setup, find_packages

with open("README.md") as f:
   long_description = f.read()

with open('requirements.txt') as f:
   requirements = f.read().splitlines()

setup(
   name="flame",
   version="0.1.0",
   license='MIT',
   description="Deep learning applied to text classification, in PyTorch",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author="John Jung",
   author_email="tojohnjung@gmail.com",
   url="https://github.com/zjohn77/flame",
   packages=find_packages(),
   classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
   ],
   keywords='deep learning',
   install_requires=requirements
)