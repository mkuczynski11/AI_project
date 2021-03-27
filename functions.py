import numpy as np
import pandas as pd

#returns top movies based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
def top_movies_general(df:pd.DataFrame, top:float) -> pd.DataFrame:
    C = float(df['vote_average'].mean())
    m = df['vote_count'].quantile(1 - top)
    
    top_movies = df[(df['vote_count'] >= m)]
    top_movies['WR'] = top_movies.apply(weighted_rating, args=(m,C,), axis=1)
    top_movies = top_movies.sort_values("WR", ascending=False)
    return top_movies

#returns top movies with specified genre based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
#genre - string which represents genre that should be listed
def top_movies_by_genre(df:pd.DataFrame, top:float, genre:str) -> pd.DataFrame:
    s = df.apply(lambda df: pd.Series(df['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = df.drop('genres', axis=1).join(s)
    top_movies = gen_md[gen_md['genre'] == genre]
    return top_movies_general(top_movies,top)

def weighted_rating(df:pd.DataFrame, m:float, C:float) -> float:
    v = df['vote_count']
    R = df['vote_average']
    return (((v/(v+m)) * R) + ((m/(m+v)) * C))


def top_movies_by_year(df: pd.DataFrame, top: float, year: int, below: bool) -> pd.DataFrame:
    top_movies = df
    
    if below:
        top_movies = top_movies[(top_movies['release_date'] <= year)]
    else:  
        top_movies = top_movies[(top_movies['release_date'] >= year)]

    return top_movies_general(top_movies, top)

        