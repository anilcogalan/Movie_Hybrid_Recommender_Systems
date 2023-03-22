import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#############################################
# 1.Data Preparation
#############################################

df_movie_ = pd.read_csv('datasets/movie.csv')
movie = df_movie_.copy()

df_rating_ = pd.read_csv('datasets/rating.csv')
rating = df_rating_.copy()

df_ = rating.merge(movie, how='left', on='movieId')
df = df_.copy()

total_user_rating = pd.DataFrame(df['title'].value_counts())
rare_movies = total_user_rating[total_user_rating['title'] <= 1000].index
rare_movies.shape

common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape

user_movie_df = common_movies.pivot_table(index=["userId"],
                                          columns=["title"],
                                          values="rating")

#############################################
# 2.Determining the Movies Watched by the User to Suggest
#############################################

# A random user id
random_user = 138491

# user ids'
user_movie_df.index[0:5]

# The user_movie_df created dataset
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()

# The reason for choosing columns is movie names
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# movies_watched: list of movies watched by random user
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

#############################################
# 3.Accessing Data and Ids of Other Users Watching the Same Movies
#############################################

# movies_watched_df.T : to the userid column, movies to the row
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.head()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Calculate the number of movies that will be the threshold value
len(movies_watched)
perc = len(movies_watched) * 60 / 100

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# Number of people similar to random user
len(users_same_movies)

#############################################
# 4.Determining the Users to be Suggested and Most Similar Users
#############################################

# Let's filter the users who are not similar to the user selected
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.shape

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Correlation of the random user we chose and other audiences
corr_df[corr_df["user_id_1"] == random_user]

# Ranking of the selected random user and the users with a correlation above 0.65
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# random user's own id from data
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# user'ids of similar trackers
top_users_ratings["userId"].unique()

#############################################
# 5.Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * \
                                       top_users_ratings['rating']
top_users_ratings.head()

# singularization of movies
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Movies with a weighted rating greater than 3.5 in recommendation_df
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5]. \
                             sort_values("weighted_rating", ascending=False)[0:5]
movies_to_be_recommend.head()

# Names of 5 recommended movies
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]

#############################################
#  6.Item-Based Recommendation
#############################################

user = 108170

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# The movie with the most up-to-date score among the movies that
# the user to be recommended gives 5 points
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
               sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Filtering user_movie_df dataframe by selected movie id
user_movie_df[movie[movie["movieId"] == movie_id]["title"]]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Correlation of selected movie and other movies using filtered dataframe
sorted_recommender = user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# 5 movie suggestions except itself
sorted_recommender[1:6].index
