from ast import literal_eval

from pandas.core.arrays.categorical import contains
from pandas.io.formats.format import common_docstring
from functions import discard_keywords, get_director, top_movies_general, top_movies_by_genre, weighted_rating, top_movies_by_year, get_actors
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer

def main():
    #----------------Data prep----------------#
    #read csv
    csv_data = pd.read_csv("movies_metadata.csv")
    #select columns, that we are interested in
    movies_data = csv_data[["genres", "vote_count", "vote_average","release_date","title"]]
    #print(movies_data)

    #First we need to fix genres column in a way that it contains arrays of each genre name
    movies_data['genres'] = movies_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    #print(movies_data)

    #Next step is to fix vote_count column and vote_average column
    movies_data['vote_count'] = movies_data['vote_count'].fillna(0).astype(int)
    #print(movies_data['vote_count'])

    # Changing release_data to year only
    movies_data = movies_data[movies_data['release_date'].notna()]
    movies_data = movies_data.reset_index(drop=True)
    movies_data['release_date'] = movies_data['release_date'].apply(lambda date: date[:4:])
    movies_data['release_date'] = pd.to_numeric(movies_data['release_date'])
    # print(movies_data['release_date'])
    #----------------Data prep----------------#

    #----------------Raw data recommending----------------#
    #First part is recommender based on raw movie ratings
    # top_general = top_movies_general(movies_data, 0.01)
    #print(top_general)

    # top_by_genre = top_movies_by_genre(movies_data, 0.05, 'Romance')
    #print(top_by_genre)

    # top_by_year = top_movies_by_year(movies_data, 0.01, 2013, below=False)
    # print(top_by_year)
    #----------------Raw data recommending----------------#

    #----------------Content based----------------#
    #--------Data prep--------#
    content_data = movies_data[['genres','release_date','title']].join(csv_data['id'])
    credits = pd.read_csv('credits.csv')
    keywords = pd.read_csv('keywords.csv')
    links = pd.read_csv('links.csv')
    links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

    #First we prep our ids in case to compare them and merge
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    content_data['id'] = content_data['id'].apply(lambda x: x if '-' not in x else -1).astype('int')

    #Merging the data
    content_data = content_data.merge(keywords, on='id')
    content_data = content_data.merge(credits, on='id')

    #Discarding unwanted data
    content_data = content_data[content_data['id'].isin(links)]

    #Prep of the crew and cast
    content_data['crew'] = content_data['crew'].apply(literal_eval).apply(lambda x: get_director(x) if isinstance(x, list) else [])
    content_data['cast'] = content_data['cast'].apply(literal_eval).apply(lambda x: get_actors(x) if isinstance(x, list) else []) 
    content_data['keywords'] = content_data['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    #print(content_data)

    keywords_frequency = content_data.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    keywords_frequency.name = 'keyword'
    keywords_frequency = keywords_frequency.value_counts()
    keywords_frequency = keywords_frequency[keywords_frequency > 1]

    stemmer = SnowballStemmer('english')
    content_data['keywords'] = content_data['keywords'].apply(discard_keywords, args=(keywords_frequency,))
    content_data['keywords'] = content_data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    content_data['keywords'] = content_data['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    content_data['soup'] = content_data['crew'] + content_data['cast'] + content_data['keywords'] + content_data['genres']
    content_data['soup'] = content_data['soup'].apply(lambda x: ' '.join(x))
    #--------Data prep--------#

    #----------------Content based----------------#

    #----------------Colaborative based----------------#

    #----------------Colaborative based----------------#
    return

if __name__ == '__main__':
    main()