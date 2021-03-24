import numpy as np
import pandas as pd

#returns top movies based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
def top_movies_general(x:pd.DataFrame, top:float) -> pd.DataFrame:
    #Computing mean ov vote average
    C = float(x['vote_average'].mean())
    #Computing top 5% movies in terms of vote casted
    m = x['vote_count'].quantile(1 - top)

    #Creating topMovies dataframe which contains movies that are in top 5% in terms of vote casted
    topMovies = x[(x['vote_count'] >= m)]
    #Creating new column for Weighted Rating
    #axis = 1 -> applying function on each row
    topMovies['WR'] = topMovies.apply(weighted_rating, args=(m,C,), axis=1)
    #amount - determines how many top movies we want to list
    #Sorting values by Weighted Rating in top movies list and getting only specified amount of them
    topMovies = topMovies.sort_values("WR", ascending=False)
    return topMovies

#returns top movies with specified genre based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
#genre - string which represents genre that should be listed
def top_movies_genre(x:pd.DataFrame, top:float, genre:str) -> pd.DataFrame:
    #Creating topMovies dataframe which contains movies that contains desired genre
    
    #return top_movies_general(topMovies,top)
    return

def weighted_rating(x:pd.DataFrame, m:float, C:float):
    v = x['vote_count']
    R = x['vote_average']
    return (((v/(v+m)) * R) + ((m/(m+v)) * C))