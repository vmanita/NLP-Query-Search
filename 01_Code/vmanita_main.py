###############################################################################
# Import libraries
###############################################################################
import os
# set path
main_path = r'C:\Users\vmanita\Documents\tdx'
os.chdir(main_path)
from vmanita_utils import load_processed_files, identity_tokenizer, guugle_search_engine
###############################################################################
# Guugle_search Engine 
###############################################################################
    
# All the calculations needed to achieve the processed documents of the news file, 
# vectorizer and tf-idf matrix are shown and commented in the 'manita_utils.py' script
# To optimize testing and running time I saved them into files that I could later load
# into the main script

news, p_news, vectorizer, tfidf_matrix = load_processed_files(os.path.join(main_path,'Processed_files'))

# Simple Queries 

# 1) Trump

free_text = 'Trump'
title = None
publication =  None
author =  None
content = None

result = guugle_search_engine(news,
                             p_news,
                             vectorizer,
                             tfidf_matrix,
                             free_text,
                             title,
                             publication, 
                             author, 
                             content)

########################## OTHER QUERIES ########################################
'''
# 2) turtle fossil

free_text = 'turtle fossil'
title = None
publication =  None
author =  None
content = None

result = guugle_search_engine(news,
                             p_news,
                             vectorizer,
                             tfidf_matrix,
                             free_text,
                             title,
                             publication, 
                             author, 
                             content)

# 3) United States of America

free_text = 'United States of America'
title = None
publication =  None
author =  None
content = None

result = guugle_search_engine(news,
                             p_news,
                             vectorizer,
                             tfidf_matrix,
                             free_text,
                             title,
                             publication, 
                             author, 
                             content)

# Complex Queries

# 1) title: Chicago

free_text = None
title = 'Chicago'
publication =  None
author =  None
content = None

result = guugle_search_engine(news,
                             p_news,
                             vectorizer,
                             tfidf_matrix,
                             free_text,
                             title,
                             publication, 
                             author, 
                             content)

# 2) title: brazil & publication: breitbart & author: frances

free_text = None
title = 'brazil'
publication =  'breitbart'
author =  'frances'
content = None

result = guugle_search_engine(news,
                             p_news,
                             vectorizer,
                             tfidf_matrix,
                             free_text,
                             title,
                             publication, 
                             author, 
                             content)

# 3) title:death penalty & content: boston

free_text = None
title = 'Death penalty'
publication =  None
author =  None
content = 'boston'

result = guugle_search_engine(news,
                             p_news,
                             vectorizer,
                             tfidf_matrix,
                             free_text,
                             title,
                             publication, 
                             author, 
                             content)

'''















