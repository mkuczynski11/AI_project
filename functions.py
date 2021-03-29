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

#returns top movies with specified genre based only on user votes
#x - DataFrame to analyze
#top - float value from 0 to 1 that determines percentage of top movies in terms of votes casted
#year - year for all movies to be compared with
#below - if other movies year values should be below the year argument
def top_movies_by_year(df: pd.DataFrame, top: float, year: int, below: bool) -> pd.DataFrame:
    top_movies = df
    
    if below:
        top_movies = top_movies[(top_movies['release_date'] <= year)]
    else:  
        top_movies = top_movies[(top_movies['release_date'] >= year)]

    return top_movies_general(top_movies, top)

#returns [director, director] from crew members list
#l - crew members list
def get_director(l:list) -> str:
    for x in l:
        if x['department'] == "Directing":
            return [str.lower(x['name'].replace(" ","")) for i in range(2)]
    return []

#returns 0:3 actors from cast members list
#l - cast members list
def get_actors(l:list) -> list:
    return [str.lower(l[i]['name'].replace(" ", "")) for i in range(3)] if len(l) > 3 else [str.lower(x['name'].replace(" ", "")) for x in l]

#returns valid keywords(more appearences than 1)
#l - list of keywords
#s - list of valid keywords
def discard_keywords(l:list, s:pd.Series) -> list:
    keywords = []
    for i in l:
        if i in s:
            keywords.append(i)
    return keywords

#returns recommendation for a certain movie based on the movies parameters
#title - movie title to check similarity
#df - movies dataframe
#cosine_sim - computed already cosine similarity matrix
def get_recommendation(title:str, df:pd.DataFrame, cosine_sim:np.ndarray) -> pd.DataFrame: # to merge
    indices = pd.Series(df.index, index=df['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

#returns recommendation for a certain movie based on the movies parameters and then sorts them by Weightened Rating
#title - movie title to check similarity
#df - movies dataframe
#cosine_sim - computed already cosine similarity matrix
def get_popular_recomandation(title:str, df:pd.DataFrame, cosine_sim:np.ndarray) -> pd.DataFrame: # to merge
    titles = get_recommendation(title, df, cosine_sim).head(100)
    top_movies = top_movies_general(titles, 0.1)
    return top_movies

# Returns DataFrame containing 100 most similiar movies, ordered by their rating
def get_recommended_movies(cos_sim: np.ndarray, titles: pd.DataFrame, df:pd.Series, title: str) -> pd.DataFrame: # to merge
    id = titles[title]

    similiar_movies = np.asarray(list(enumerate(cos_sim[id])))
    similiar_movies = similiar_movies[np.argsort(-1 * similiar_movies[:,1])]
    similiar_movies = similiar_movies[:101:]
    movie_slice = [x[0] for x in similiar_movies]

    recommended = df.iloc[movie_slice]

    top_recommended = top_movies_general(recommended, 0.1)

    return top_recommended

# Displays top recommended movies in a pleasant way
def view_recommended_movies(recommended: pd.DataFrame) -> None:     #to change 
    print('=' * 60)
    print(f"Top recommended movies for {recommended['title'][0]}:")
    i = 1
    for x in recommended['title'].values.tolist():
        print(f'{i}. {x}')
        i+=1

    print('=' * 60)

def save_to_file(df: pd.DataFrame, overwrite=True) -> None:
    pass

def load_from_file(file_name: str) -> pd.DataFrame:
    pass
