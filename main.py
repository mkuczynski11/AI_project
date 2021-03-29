from ast import literal_eval

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             _analyze)
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

pd.options.mode.chained_assignment = None  # default='warn' <- disabling warning message

from functions import (get_recommended_movies, top_movies_by_genre,
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
    links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

    print("Recommender: content_data preping")
    #content_data DataFrame prep
    content_data:pd.DataFrame = movies_data[['genres','release_date','title', 'vote_count', 'vote_average']].join(csv_data['id'])
    content_data['id'] = content_data['id'].apply(lambda x: x if '-' not in x else -1).astype('int')
    
    print("Recommender: content_data merging")
    #Merging the data with keywords and credits
    content_data = content_data.merge(keywords, on='id')
    content_data = content_data.merge(credits, on='id')
    #Discarding unwanted data
    content_data = content_data[content_data['id'].isin(links)]

    print("Recommender: content_data refactoring")
    #Prep of the crew and cast
    content_data['crew'] = content_data['crew'].apply(literal_eval).apply(lambda x: get_director(x) if isinstance(x, list) else [])
    content_data['cast'] = content_data['cast'].apply(literal_eval).apply(lambda x: get_actors(x) if isinstance(x, list) else []) 
    content_data['keywords'] = content_data['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    print("Recommender: keywords discarding")
    #Discarding unwanted keywords(with few occurences)
    keywords_frequency:object = content_data.apply(lambda x: pd.Series(x['keywords'], dtype="object"), axis=1).stack().reset_index(level=1, drop=True)
    keywords_frequency.name = 'keyword'
    keywords_frequency = keywords_frequency.value_counts()
    keywords_frequency = keywords_frequency[keywords_frequency > 1]
    
    print("Recommender: keywords refactoring")
    #Refactoring keywords with stemmer(converts various keywords into one)
    stemmer = SnowballStemmer('english')
    content_data['keywords'] = content_data['keywords'].apply(discard_keywords, args=(keywords_frequency,))
    content_data['keywords'] = content_data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    content_data['keywords'] = content_data['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    print("Recommender: cosine similarity computing")
    #Cosine similarity computing
    content_data['soup'] = content_data['crew'] + content_data['cast'] + content_data['keywords'] + content_data['genres']
    content_data['soup'] = content_data['soup'].apply(lambda x: ' '.join(x))
    count = CountVectorizer(analyzer="word", ngram_range=(1,2), min_df=0, stop_words="english")
    count_matrix = count.fit_transform(content_data['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
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
    # Geting new DataFrame with specific columns and creating additional column
    # with tagline and overview combined
    movies_df = pd.read_csv("movies_metadata.csv", low_memory=False)[["genres", "vote_count", "vote_average", "release_date", "title", "tagline", "overview"]]
    movies_df['tagline'] = movies_df['tagline'].fillna('')
    movies_df['description'] = movies_df['tagline'] + movies_df['overview']
    movies_df['description'] = movies_df['description'].fillna('')
    # print("DataFrame with description column:")
    # print(md_taglined)
    movies_df = movies_df.reset_index()

    # Creating tf-idf statistics
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_values = tf.fit_transform(movies_df['description'])
    # print(tfidf_values.shape)

    cosine_simil = linear_kernel(tfidf_values, tfidf_values)
    # print(cosine_similarity)

    titles = pd.Series(movies_df.index, index=movies_df['title'], name="titles")
    # print(movies)

    # titles = md_taglined['title']
    # print(titles)

    recommended = get_recommended_movies(cosine_simil, titles, movies_df, 'Toy Story')
    view_recommended_movies(recommended)
    #----------------Content description based----------------#


    #----------------Content soup based----------------#

    # content_recommend = get_recommendation('Toy Story', content_data, cosine_sim).head(15)
    # print(content_recommend)
    print("Recommender: popular_content_recommending preping")
    popular_content_recommend = get_popular_recomandation('The Wolf of Wall Street', content_data, cosine_sim)
    print(popular_content_recommend)

    #----------------Content soup based----------------#



    #------------Colaborative based-----------#

    #------------Colaborative based-----------#
    return

if __name__ == '__main__':
    main()
