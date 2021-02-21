# Lexical-Simplification

## Download the wikihow dataset

```bash
wget https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/complex-word-identification-dataset/cwishareddataset.zip
```

## Unzip data as dataset folder
```bash
unzip cwishareddataset.zip -d datset
```

## Download the glove model
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
```

## Unzip model as embeddings folder
```bash
unzip glove.6B.zip -d embeddings
```

## install require python libraries
```bash
pip3 install nltk pandas matplotlib keras gensim sklearn seaborn
```