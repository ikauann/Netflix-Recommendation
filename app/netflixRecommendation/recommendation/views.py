from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponse

import pandas            as pd
from sklearn.metrics.pairwise          import cosine_similarity
from sklearn.feature_extraction.text   import TfidfVectorizer

df = pd.read_csv(r'E:\Github\netflix recommendation\netflix_titles.csv')

def recommendation(title):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['description'])
    vectorizer.get_feature_names_out()  

    cos_sim = cosine_similarity(X, X)
    indices = pd.Series(df.index, df['title'])

    i = indices[title]
    list_similarity = list(enumerate(cos_sim[i]))
    ordenado = sorted(list_similarity, key=lambda x: x[1], reverse=True)
    top_10 = ordenado[0:21]
    movie_index = [movie[0] for movie in top_10]
    return [movie for movie in df['title'].iloc[movie_index].values]

def index(request):
    return HttpResponse(recommendation('Stranger Things'))
