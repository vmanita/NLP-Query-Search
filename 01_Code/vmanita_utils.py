###############################################################################
# Import libraries
###############################################################################
import nltk
import pandas as pd
import numpy as np
import re
import math   
import string
from nltk.tokenize import word_tokenize as tknzr
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from collections import defaultdict
from collections import Counter
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import scipy.sparse
import pickle
import os
from pyfiglet import Figlet
###############################################################################
# Import data
###############################################################################
# Importing the data and dropping irrelevant column

#path = r'C:\Users\vmanita\Documents\tdx'
#news_path = os.path.join(path, 'articles1.csv')
#news = pd.read_csv(news_path)
#news.drop(columns = 'Unnamed: 0',inplace = True)
#news.to_json(os.path.join(path, 'Processed_files\original_text.json'))


def load_processed_files(path):
    vectorizer = pickle.load(open(os.path.join(path, 'vectorizer.pkl'), "rb" ) )
    news = pd.read_json(os.path.join(path, 'original_text.json'))
    p_news = pd.read_json(os.path.join(path, 'processed_news.json'))
    tfidf_matrix = scipy.sparse.load_npz(os.path.join(path, 'tfidf_matrix.npz'))
    
    return news, p_news, vectorizer, tfidf_matrix
###############################################################################
# Process data
###############################################################################

# Processing the text
# 1) Tokenize and lowercase
# 2) Remove punctuation and special characters
# 3) Remove stopwords
# 4) Apply stemming
# 5) Removing blank/empty strings and single characters from analysis

stopwords = nltk.corpus.stopwords.words('english')
word_lemma = WordNetLemmatizer()
word_stem = EnglishStemmer()

def process_text(text):
    punctuation_to_remove = string.punctuation + "’‘—“”"
    strip = str.maketrans('', '', punctuation_to_remove)
    sub_filter = r"\b[a-zA-Z]\b"
    p_text = list(filter(None, [re.sub(sub_filter, "", word_stem.stem(word.translate(strip)))for 
                                word in tknzr(text.lower()) 
                                if word not in stopwords]))
    return p_text

###############################################################################
# Inverted Indexing
###############################################################################

# Process text column in the News dataset, We will do this once and save it in a excel file
# so we can later test without having to process the 50 000 records all over again
'''    
p_news = news.copy()[['id','title','publication','author','content']]

for col in p_news.columns:
    if col != 'id':
        p_news[col] = [process_text(str(x)) for x in news[col]]
        
p_news['text'] = p_news['title'] + p_news['content']

p_news.to_json(os.path.join(path, 'processed_news.json'))
'''

# Import processed news documents text
#p_news = pd.read_json(os.path.join(path, 'processed_news.json'))

#Create an inverted index search that returns, for a specified token, the ID of 
#the documents where that token appears 

def create_index(text):
    index = defaultdict(list)
    for i, tokens in enumerate(text):
        for token in tokens:
            index[token].append(i)
    return index

#teste_idx = create_index(p_news['text'])

def SimpleComplexSearch(text, config):  
    
    free_text = config['free_text']
    title = config['title']
    publication = config['publication']
    author = config['author']
    content = config['content']
  
    # Simple search: Title and Text
    search_docs = {}
    if free_text:
        docs_id = create_index(text['text'])
        search_docs['free_text'] = []
        for word in free_text:
                search_docs['free_text'] += list(set(docs_id[word]))
    
    # Complex search: Specific fields (< & > operator)            
    else:
        
        if title:
            docs_id = create_index(text['title'])
            search_docs['title'] = []
            for word in title:
                search_docs['title'] += list(set(docs_id[word]))
                
        if publication:
            docs_id = create_index(text['publication'])
            search_docs['publication'] = []
            for word in publication:
                search_docs['publication'] += list(set(docs_id[word]))
    
        if author:
            docs_id = create_index(text['author'])
            search_docs['author'] = []
            for word in author:
                search_docs['author'] += list(set(docs_id[word]))
                
        if content:
            docs_id = create_index(text['content'])
            search_docs['content'] = []
            for word in content:
                search_docs['content'] += list(set(docs_id[word]))
        
    search_docs = list(reduce(set.intersection, (set(val) for val in search_docs.values())))        
    print('================================================')
    print('>>> Found {} news with the desired keywords'.format(len(search_docs)))
    print('================================================')
    return search_docs

###############################################################################
# TF IDF Matrix
###############################################################################

def identity_tokenizer(text):
    return text
# *******************
#  Vectorize
# *******************
    
#vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words=None, lowercase=False)   
# Export vectorizer > Pickle
#pickle.dump(vectorizer,open(os.path.join(path, "vectorizer.pkl"),"wb"))
# Load the content
#vectorizer = pickle.load(open(os.path.join(path, "vectorizer.pkl"), "rb" ) )

# *******************
#  TFIDF Matrix
# *******************
#tfidf_matrix = vectorizer.fit_transform(p_news['text'])
# Export TFIDF Matrix > npz
#scipy.sparse.save_npz(os.path.join(path, 'tfidf_matrix.npz'), tfidf_matrix)
# load the sparse matrix
#tfidf_matrix = scipy.sparse.load_npz(os.path.join(path, 'tfidf_matrix.npz'))

###############################################################################
# Search Engine
###############################################################################

def guugle_search_engine(o_text, text, vectorizer, tfidf_matrix, free_text = None, title = None, publication = None, author = None, content = None, top = 20):
    custom_fig = Figlet(font='big')
    print(custom_fig.renderText('Guugle Search'))
    # 2) Process Query
    config = {'free_text': free_text,
              'title': title, 
              'publication': publication, 
              'author': author, 
              'content': content}

    p_query = []
    for key, value in config.items():
        if value:
            config[key] = process_text(str(value))
            if free_text:
                # query is free_text
                p_query = config[key]
                break
            else:
                # Add keywords to query if specific fields
                p_query.extend(config[key])
            
    # 3) Get all documents with query
    search_docs = SimpleComplexSearch(text, config)
    
    # 4) Vectorize query
    query_tfidf = vectorizer.transform([p_query])
   
    # 5) Cosine similarity
    cosineSimilarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    result = pd.DataFrame({'CosSimScore':np.round(cosineSimilarities, 3)}).sort_values(by = 'CosSimScore', ascending = False)
    
    # 6) Cross reference step 2) with step 5)
    result = result.iloc[result.index.isin(search_docs)]
    
    # 7) Order by rank and return documents
    result['rank'] = result['CosSimScore'].rank(ascending=False).astype(int)
    result = result[:top]
    result['news_id'] = text.iloc[result.index]['id'].values
    result = result.merge(o_text[['id','title','content']], left_on='news_id', right_on='id')
    print(result[['CosSimScore','rank','news_id']])
    return result



















