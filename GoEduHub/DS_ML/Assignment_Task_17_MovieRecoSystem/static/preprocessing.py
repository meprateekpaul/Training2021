import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df2 = pd.read_csv('movie_metadata.csv')
dt=df2.drop([ 'color', 'num_critic_for_reviews',
       'duration', 'director_facebook_likes', 'actor_3_facebook_likes',
        'actor_1_facebook_likes', 'gross',
       'num_voted_users',
       'cast_total_facebook_likes', 'facenumber_in_poster',
       'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews', 'language',
       'country', 'content_rating', 'budget', 'title_year',
       'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
       'movie_facebook_likes'],axis=1)
dt.dropna(inplace=True)
dt['genres']=dt['genres'].apply(lambda a: str(a).replace('|',' '))
dt['movie_title']=dt['movie_title'].apply(lambda a: a[:-1])
dt['combined']=dt['director_name']+' ' +dt['actor_2_name']+' '+dt['actor_1_name']+ ' ' +dt['genres']+' '+dt['actor_3_name']
dt.to_csv('movie_finalized_data.csv')