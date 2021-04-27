from ast import literal_eval

import pandas as pd
from InquirerPy import inquirer
from nltk.stem.snowball import SnowballStemmer
from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

pd.options.mode.chained_assignment = None  # default='warn' <- disabling warning message

from functions import (discard_keywords, get_actors, get_director,
                       get_popular_recomandation, get_recommendation,
                       get_recommended_movies, hybrid_recommendation,
                       top_movies_by_genre,
                       top_movies_by_year, top_movies_general, top_recommended_movies_for_user,
                       view_recommended_movies, weighted_rating)


def main():
    #----------------Data prep----------------#
    print("Recommender: reading csv")
    #read csv
    csv_data = pd.read_csv("movies_metadata.csv", low_memory=False)

    print("Recommender: movies_data preping")
    #movie_data DataFrame columns selecting
    movies_data:DataFrame = csv_data[["id","genres", "vote_count", "vote_average","release_date","title"]]
    #movie_data title column unique
    movies_data = movies_data.drop_duplicates(subset=['title'])
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
    content_data_soup:pd.DataFrame = movies_data[['id','genres','release_date','title', 'vote_count', 'vote_average']]
    # content_data_soup = top_movies_general(content_data_soup, 0.7)
    # content_data_soup['id'] = csv_data['id']
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
    # Discarding unwanted keywords(with few occurences)
    keywords_frequency:object = content_data_soup.apply(lambda x: pd.Series(x['keywords'], dtype="object"), axis=1).stack().reset_index(level=1, drop=True)
    keywords_frequency.name = 'keyword'
    keywords_frequency = keywords_frequency.value_counts()
    keywords_frequency = keywords_frequency[keywords_frequency > 1]
    
    print("Recommender: keywords refactoring")
    # Refactoring keywords with stemmer(converts various keywords into one)
    stemmer = SnowballStemmer('english')
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(discard_keywords, args=(keywords_frequency,))
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    content_data_soup['keywords'] = content_data_soup['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    print("Recommender: cosine similarity computing(soup)")
    # Cosine similarity computing
    content_data_soup['soup'] = content_data_soup['crew'] + content_data_soup['cast'] + content_data_soup['keywords'] + content_data_soup['genres']
    content_data_soup['soup'] = content_data_soup['soup'].apply(lambda x: ' '.join(x))
    count = CountVectorizer(analyzer="word", ngram_range=(1,2), min_df=0, stop_words="english")
    count_matrix = count.fit_transform(content_data_soup['soup'])
    cosine_sim_soup = cosine_similarity(count_matrix, count_matrix)

    print("Recommender: description prep")
    # Geting new DataFrame with specific columns and creating additional column
    # with tagline and overview combined
    content_data_desc = csv_data[['id','genres','release_date','title', 'vote_count', 'vote_average', 'overview', 'tagline']]

    # Trimming dataset so there will be only best 70% movies from this dataset
    content_data_desc = top_movies_general(content_data_desc, 0.7)
    content_data_desc['tagline'] = content_data_desc['tagline'].fillna('')
    content_data_desc['description'] = content_data_desc['tagline'] + content_data_desc['overview']
    content_data_desc['description'] = content_data_desc['description'].fillna('')
    content_data_desc = content_data_desc.reset_index()
    content_data_desc['id'] = content_data_desc['id'].apply(lambda x: x if '-' not in x else -1).astype('int')

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

    # #----------------Content description based----------------#
    # movie = 'Star Wars'
    # print("Recommender: recommending description based for " + movie)
    # recommended = get_popular_recomandation(movie, content_data_desc, cosine_sim_desc)
    # view_recommended_movies(recommended)
    # #----------------Content description based----------------#

    #--------------------Content soup based-------------------#
    # movie = 'Star Wars'
    # print("Recommender: content_recommending preping for " + movie)
    # content_recommend = get_recommendation(movie, content_data_soup, cosine_sim_soup).head(15)
    # view_recommended_movies(content_recommend)
    # print("Recommender: popular_content_recommending preping for " + movie)
    # popular_content_recommend = get_popular_recomandation(movie, content_data_soup, cosine_sim_soup)
    # view_recommended_movies(popular_content_recommend)
    #--------------------Content soup based-------------------#

    content_data = content_data_soup
    cosine_sim = cosine_sim_soup

    #------------User preparation-----------#
    ratings:DataFrame = read_csv("ratings_small.csv")
    users_count = 1
    movies_to_rate_count = 10

    for i in range(users_count):
        print(f"Wybierz {movies_to_rate_count} filmy dla użytkownika {i+1}")
        user_id = 1000 + i+1
        for i in range(movies_to_rate_count):
            user_choice = inquirer.select(
                message="Wybierz film",
                choices=top_movies_general(content_data,0.05)['title']
            ).execute()

            user_rating = inquirer.select(
                message="Podaj ocene filmu",
                choices=[i for i in range(1,6)]
            ).execute()

            tmp_id = content_data[(content_data['title'] == user_choice)]['id']
            tmp_id = int(links[(links['tmdbId'] == int(tmp_id))]['movieId'])
            d = {'userId' : [user_id], 'movieId' : [tmp_id], 'rating' : [user_rating], "timestamp" : [-1]}
            tmp_df = pd.DataFrame(data=d)
            ratings = ratings.append(tmp_df, ignore_index=True)

    #------------User preparation-----------#

    #------------Colaborative based-----------#
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    algo = SVD()
    # algo = KNNWithMeans()
    print("Recommender: evaluating 'RMSE' and 'MAE' measures for SVD")
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    #------------Colaborative based-----------#

    # #------------Hybrid recommender-----------#
    # print("Recommender: Hybrid recommendation for " + movie)
    # hybrid_result = hybrid_recommendation(movie, 1, content_data_soup, cosine_sim_soup, svd, links)
    # print(hybrid_result[['title', 'est']])
    # hybrid_result = hybrid_recommendation(movie, 500, content_data_soup, cosine_sim_soup, svd, links)
    # print(hybrid_result[['title', 'est']])
    # #------------Hybrid recommender-----------#

    #------------CLI--------------------------#
    print('Welcome to the movies recommender system, please choose one of the following:')

    recommendation_type = 'Hybrid'

    while True:

        choice = inquirer.select(
            message='Actions: ', 
            choices=['Search for the movie',
                     'Choose recommendation type',
                     'Print top 1 percent movies',
                     'Top For User',
                     'Find movie',
                     'Exit application']
        ).execute()

        if choice == 'Exit application':
            print('Thank you for the effort')
            break

        if choice == 'Find movie':
            movie_title = inquirer.text(message='Please, enter movie name:').execute()
            titles = []
            for i in range(len(content_data)):
                if(movie_title in content_data['title'].iloc[i]):
                    titles.append(content_data['title'].iloc[i])
            print(titles)

        if choice == 'Top For User':
            user_choice = inquirer.select(
                message='Select user', 
                choices=[i+1001 for i in range(users_count)]
            ).execute()
            movies = top_recommended_movies_for_user(user_choice, content_data, algo, links).head(10)
            print(movies[['title', 'est']])


        if choice == 'Print top 1 percent movies':
            view_recommended_movies(top_movies_general(content_data, 0.01))

        if choice == 'Search for the movie':
            user_choice = inquirer.select(
                message='Select user', 
                choices=[i+1001 for i in range(users_count)]
            ).execute()

            movie_title = inquirer.text(message='Please, enter movie name:').execute()

            print(f"It seems that you are looking for recommendations for {movie_title}")
            tmp_title = movies_data[(movies_data['title'] == movie_title)]
            if(tmp_title.empty):
                print(f"There is no movie named {movie_title}")
                continue
            elif(recommendation_type == 'Hybrid'):
                result = hybrid_recommendation(movie_title, user_choice,
                content_data, cosine_sim, algo, links)
            elif(recommendation_type == 'Content based'):
                result = get_popular_recomandation(movie_title, content_data, cosine_sim)
            elif(recommendation_type == 'Colaborative'):
                print("Not yet implemented")

            print(f"Your recommendations for {movie_title}")
            # view_recommended_movies(result)
            print(result[['title','est']])
            chosen_movie = inquirer.select(
                message='Please, select the movie you want to rate',
                choices=[x for x in result['title']]
            ).execute()
            rating = inquirer.select(
                message='Please rate the movie',
                choices=[i for i in range(1,6)]
            ).execute()

            tmp_id = content_data[(content_data['title'] == chosen_movie)]['id']
            tmp_id = int(links[(links['tmdbId'] == int(tmp_id))]['movieId'])
            d = {'userId' : [user_id], 'movieId' : [tmp_id], 'rating' : [rating], "timestamp" : [-1]}
            tmp_df = pd.DataFrame(data=d)
            ratings = ratings.append(tmp_df, ignore_index=True)

            data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
            trainset = data.build_full_trainset()
            algo.fit(trainset)


        if choice == 'Choose recommendation type':
            recommendation_type = inquirer.select(
                message='Please, choose recommendation type',
                choices=['Hybrid', 'Content description based',
                          'Colaborative']
            ).execute()
    #------------CLI--------------------------#

    return

if __name__ == '__main__':
    main()

#usunac powtorki filmow