from ast import literal_eval

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from pandas.io.parsers import read_csv
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,)
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

pd.options.mode.chained_assignment = None  # default='warn' <- disabling warning message

from functions import (get_recommended_movies, hybrid_recommendation, top_movies_by_genre,
                       top_movies_by_year, top_movies_general,
                       view_recommended_movies, weighted_rating,
                       discard_keywords, get_director, get_recommendation,
                       get_popular_recomandation, get_actors)

def main():
    #----------------Data prep----------------#
    print("Recommender: reading csv")
    #read csv
    csv_data = pd.read_csv("movies_metadata.csv", low_memory=False)

    print("Recommender: movies_data preping")
    #movie_data DataFrame columns selecting
    movies_data = csv_data[["genres", "vote_count", "vote_average","release_date","title"]]
    #movie_data genres column fixing
    movies_data['genres'] = movies_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    #movie_data vote count column fixing
    movies_data['vote_count'] = movies_data['vote_count'].fillna(0).astype(int)
    #movie_data release_date column fixing
    movies_data = movies_data[movies_data['release_date'].notna()]
    movies_data = movies_data.reset_index(drop=True)
    movies_data['release_date'] = movies_data['release_date'].apply(lambda date: date[:4:])
    movies_data['release_date'] = pd.to_numeric(movies_data['release_date'])
    
    print("Recommender: credits preping")
    #credits DataFrame prep
    credits = pd.read_csv('credits.csv')
    credits['id'] = credits['id'].astype('int')

    print("Recommender: keywords preping")
    #keywords DataFrame prep
    keywords = pd.read_csv('keywords.csv')
    keywords['id'] = keywords['id'].astype('int')

    print("Recommender: links preping")
    #links DataFrame prep
    links = pd.read_csv('links.csv')
    links['tmdbId'] = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

    print("Recommender: content_data preping")
    #content_data DataFrame prep
    # content_data_soup:pd.DataFrame = movies_data[['genres','release_date','title', 'vote_count', 'vote_average']].join(csv_data['id'])
    content_data_soup:pd.DataFrame = movies_data[['genres','release_date','title', 'vote_count', 'vote_average']]
    content_data_soup['id'] = csv_data['id']
    content_data_soup['id'] = content_data_soup['id'].apply(lambda x: x if '-' not in x else -1).astype('int')
    
    print("Recommender: content_data merging")
    #Merging the data with keywords and credits
    content_data_soup = content_data_soup.merge(keywords, on='id')
    content_data_soup = content_data_soup.merge(credits, on='id')
    #Discarding unwanted data
    content_data_soup = content_data_soup[content_data_soup['id'].isin(links['tmdbId'])]

    print("Recommender: content_data refactoring")
    #Prep of the crew and cast
    content_data_soup['crew'] = content_data_soup['crew'].apply(literal_eval).apply(lambda x: get_director(x) if isinstance(x, list) else [])
    content_data_soup['cast'] = content_data_soup['cast'].apply(literal_eval).apply(lambda x: get_actors(x) if isinstance(x, list) else []) 
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    print("Recommender: keywords discarding")
    #Discarding unwanted keywords(with few occurences)
    keywords_frequency:object = content_data_soup.apply(lambda x: pd.Series(x['keywords'], dtype="object"), axis=1).stack().reset_index(level=1, drop=True)
    keywords_frequency.name = 'keyword'
    keywords_frequency = keywords_frequency.value_counts()
    keywords_frequency = keywords_frequency[keywords_frequency > 1]
    
    print("Recommender: keywords refactoring")
    #Refactoring keywords with stemmer(converts various keywords into one)
    stemmer = SnowballStemmer('english')
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(discard_keywords, args=(keywords_frequency,))
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    print("Recommender: cosine similarity computing(soup)")
    #Cosine similarity computing
    content_data_soup['soup'] = content_data_soup['crew'] + content_data_soup['cast'] + content_data_soup['keywords'] + content_data_soup['genres']
    content_data_soup['soup'] = content_data_soup['soup'].apply(lambda x: ' '.join(x))
    count = CountVectorizer(analyzer="word", ngram_range=(1,2), min_df=0, stop_words="english")
    count_matrix = count.fit_transform(content_data_soup['soup'])
    cosine_sim_soup = cosine_similarity(count_matrix, count_matrix)

    print("Recommender: description prep")
    # Geting new DataFrame with specific columns and creating additional column
    # with tagline and overview combined
    content_data_desc = csv_data[['genres','release_date','title', 'vote_count', 'vote_average', 'overview', 'tagline']]
    content_data_desc['tagline'] = content_data_desc['tagline'].fillna('')
    content_data_desc['description'] = content_data_desc['tagline'] + content_data_desc['overview']
    content_data_desc['description'] = content_data_desc['description'].fillna('')
    content_data_desc = content_data_desc.reset_index()

    print("Recommender: cosine similarity computing(desc)")
    # Creating tf-idf statistics
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_values = tf.fit_transform(content_data_desc['description'])

    cosine_sim_desc = linear_kernel(tfidf_values, tfidf_values)

    titles = pd.Series(content_data_desc.index, index=content_data_desc['title'], name="titles")
    #----------------Data prep----------------#

    #----------Raw data recommending----------#
    # top_general = top_movies_general(movies_data, 0.01)
    # print(top_general)

    # top_by_genre = top_movies_by_genre(movies_data, 0.05, 'Romance')
    #print(top_by_genre)

    # top_by_year = top_movies_by_year(movies_data, 0.01, 2013, below=False)
    # print(top_by_year)
    #----------------Raw data recommending----------------#

    #----------------Content description based----------------#
    movie = 'Avatar'
    print("Recommender: recommending description based for " + movie)
    recommended = get_popular_recomandation(movie, content_data_desc, cosine_sim_desc)
    view_recommended_movies(recommended)
    #----------------Content description based----------------#


    #--------------------Content soup based-------------------#
    print("Recommender: content_recommending preping for " + movie)
    content_recommend = get_recommendation(movie, content_data_soup, cosine_sim_soup).head(15)
    view_recommended_movies(content_recommend)
    print("Recommender: popular_content_recommending preping for " + movie)
    popular_content_recommend = get_popular_recomandation(movie, content_data_soup, cosine_sim_soup)
    view_recommended_movies(popular_content_recommend)
    #--------------------Content soup based-------------------#



    #------------Colaborative based-----------#
    reader = Reader()
    ratings = read_csv("ratings_small.csv")
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    svd = SVD()
    print("Recommender: evaluating 'RMSE' and 'MAE' measures for SVD")
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    #------------Colaborative based-----------#

    #------------Hybrid recommender-----------#
    print("Recommender: Hybrid recommendation for " + movie)
    hybrid_result = hybrid_recommendation(movie, 1, content_data_soup, cosine_sim_soup, svd, links)
    view_recommended_movies(hybrid_result)
    hybrid_result = hybrid_recommendation(movie, 500, content_data_soup, cosine_sim_soup, svd, links)
    view_recommended_movies(hybrid_result)
    #------------Hybrid recommender-----------#
    return

if __name__ == '__main__':
    main()