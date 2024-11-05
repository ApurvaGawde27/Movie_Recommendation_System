#!/usr/bin/env python
# coding: utf-8

# In[1]:


#content based recommendation system - tags
import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head(3)


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits, on='title') #WE ARE MERGINF TWO DATASETS ON THE BASIS OF 'TITLE' COLUMN


# In[8]:


movies.head(1)


# In[6]:


#genres, id, keywords, title, overview, cast, crew are important cols. All the other cols will be removed.
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.head()


# In[7]:


movies.isnull().sum()


# In[8]:


movies.dropna(inplace=True)


# In[9]:


movies.duplicated().sum()


# In[10]:


movies.iloc[0].genres


# In[11]:


#converting the above dict in list format ['Action', 'Adventure', 'Fantasy', 'Scifi']
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj): #since the dict is string, we will use ast.literal_eval funct to convert it to int.
        L.append(i['name'])
    return L


# In[12]:


movies['genres'] = movies['genres'].apply(convert)


# In[13]:


movies.head(2)


# In[14]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[15]:


def convert1(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[16]:


movies['cast'] = movies['cast'].apply(convert1)


# In[17]:


movies.head(2)


# In[18]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[19]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[20]:


movies['overview'] = movies['overview'].apply(lambda x:x.split()) #since 'overview' is string, we'll convert it to list


# In[21]:


#erasing the space between names of a person or thing. Eg: Science fiction = Sciencefiction.
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[22]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[23]:


movies.head()


# In[24]:


#concat 4 cols into 1(tags)
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[25]:


movies.head(2)


# In[26]:


new_df = movies[['movie_id', 'title', 'tags']]
new_df.head()


# In[27]:


#convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[28]:


new_df.head(2)


# In[29]:


#uppercase to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head(2)


# In[30]:


#VECTORIZATION
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[31]:


vectors = cv.fit_transform(new_df['tags']).toarray()
vectors


# In[32]:


cv.get_feature_names()


# In[33]:


#stemming wil convert same word written in different ways into the oginal word. 
#eg : ['love', 'loved','loving'] to ['love','love', 'love']
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[34]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[35]:


new_df['tags']= new_df['tags'].apply(stem)


# In[36]:


#CALCULATNG COSINE DISTANCE B/W TWO VECTORS
from sklearn.metrics.pairwise import cosine_similarity


# In[37]:


similarity = cosine_similarity(vectors)


# In[38]:


sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])[1:6] #similarity of first movie with 6  movies


# In[39]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        print()


# In[40]:


recommend('Avatar')


# In[41]:


import pickle


# In[43]:


pickle.dump(new_df, open('movies.pkl', 'wb'))


# In[44]:


pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))


# In[45]:


pickle.dump(similarity,open('similarity.pkl','wb'))

