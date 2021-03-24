from ast import literal_eval
from functions import top_movies_general, top_movies_genre, weighted_rating
import pandas as pd
import numpy as np

def main():
    #----------------Data prep----------------#
    #read csv
    movies_data = pd.read_csv("movies_metadata.csv", dtype='unicode')
    #select columns, that we are interested in
    movies_data = movies_data[["genres", "vote_count", "vote_average","release_date","title"]]
    #print(movies_data)

    #First we need to fix genres column in a way that it contains arrays of each genre name

    #if value of genres in NA/NaN, then replace it with []
    #then we apply literal_eval to this column, wchich makes our value a(we will consider list only)
    #then we take each value x from our column and if x is list, than we take all 'name' elements from ach dictionary in list
    #if x is not list, than we simply replace any data in column with empty list
    movies_data['genres'] = movies_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    #Next step is to fix vote_count column and vote_average column

    #Getting the votes count for movies as ints
    #fillna(0) -> filling NA/NaN values with 0
    movies_data['vote_count'] = movies_data['vote_count'].fillna(0).astype(int)
    #Getting the votes average for movies as ints
    movies_data['vote_average'] = movies_data['vote_average'].astype(float)
    #print(movies_data['vote_count'])
    #print(movies_data['vote_average'])

    #----------------Data prep----------------#

    #----------------Raw data recommending----------------#
    #First part is recommender based on raw movie ratings
    top_general = top_movies_general(movies_data, 0.01)
    print(top_general)

    top_by_genre = top_movies_genre(movies_data, 0.05, 'Romance')
    print(top_by_genre)
    #----------------Raw data recommending----------------#
    return

if __name__ == '__main__':
    main()