from ast import literal_eval
from functions import top_movies_general, top_movies_by_genre, weighted_rating, top_movies_by_year
import pandas as pd
import numpy as np

def main():
    #----------------Data prep----------------#
    #read csv
    movies_data = pd.read_csv("movies_metadata.csv")
    #select columns, that we are interested in
    movies_data = movies_data[["genres", "vote_count", "vote_average","release_date","title"]]
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

    top_by_year = top_movies_by_year(movies_data, 0.01, 2013, below=False)
    print(top_by_year)
    #----------------Raw data recommending----------------#

    #----------------Content based----------------#

    #----------------Content based----------------#

    #----------------Colaborative based----------------#

    #----------------Colaborative based----------------#
    return

if __name__ == '__main__':
    main()