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

def get_director(l:list) -> str:
    for x in l:
        if x['department'] == "Directing":
            return [str.lower(x['name'].replace(" ","")) for i in range(2)]
    return []

def get_actors(l:list) -> list:
    return [str.lower(l[i]['name'].replace(" ", "")) for i in range(3)] if len(l) > 3 else [str.lower(x['name'].replace(" ", "")) for x in l]

def discard_keywords(l:list, s:pd.Series) -> list:
    keywords = []
    for i in l:
        if i in s:
            keywords.append(i)
    return keywords

def get_recommendation(title:str, df:pd.DataFrame, cosine_sim:np.ndarray) -> pd.DataFrame:
    indices = pd.Series(df.index, index=df['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def get_popular_recomandation(title:str, df:pd.DataFrame, cosine_sim:np.ndarray) -> pd.DataFrame:
    titles = get_recommendation(title, df, cosine_sim).head(100)
    top_movies = top_movies_general(titles, 0.1)
    return top_movies