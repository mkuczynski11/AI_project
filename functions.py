import numpy as np
import pandas as pd

#returns top movies based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
def top_movies_general(x:pd.DataFrame, top:float) -> pd.DataFrame:
    C = float(x['vote_average'].mean())
    m = x['vote_count'].quantile(1 - top)
    
    topMovies = x[(x['vote_count'] >= m)]
    topMovies['WR'] = topMovies.apply(weighted_rating, args=(m,C,), axis=1)
    topMovies = topMovies.sort_values("WR", ascending=False)
    return topMovies

#returns top movies with specified genre based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
#genre - string which represents genre that should be listed
def top_movies_genre(df:pd.DataFrame, top:float, genre:str) -> pd.DataFrame:
    s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = df.drop('genres', axis=1).join(s)
    topMovies = gen_md[gen_md['genre'] == genre]
    return top_movies_general(topMovies,top)

def weighted_rating(x:pd.DataFrame, m:float, C:float):
    v = x['vote_count']
    R = x['vote_average']
    return (((v/(v+m)) * R) + ((m/(m+v)) * C))