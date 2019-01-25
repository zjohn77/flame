from setuptools import setup, find_packages

with open("README.md") as f:
   long_description = f.read()

setup(
   name="inflame",
   version="1.0.1",
   license='MIT',
   description="Convolutional Neural Networks--made easy to reapply to new problems",
   long_description=long_description,
   long_description_content_type="text/markdown",
   url="https://github.com/zjohn77/inflame",
   author="John Jung",
   author_email="tojohnjung@gmail.com",
   classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: POSIX :: Linux",
   ],
   keywords='deep-learning',
   install_requires=[
      'scipy>=1.1',
      'numpy>=1.15',
      'torch>=0.4',
      'scikit_learn>=0.20',
      'gensim>=3',
      'corpus4classify>=0.1'
   ],
   packages=find_packages(),
   entry_points = {
      "console_scripts": [
         "inflame_run = inflame.__main__:main"
      ]
   }
)