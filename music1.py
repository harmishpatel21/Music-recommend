# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 19:38:37 2018

@author: harry
"""

# import learn as learn
from asyncio import sleep
import pandas
# import sklearn
import scipy
from sklearn.cross_validation import train_test_split
import RecommenderSystem

triplets_file = 'E:/everything/datasets/a.txt'
songs_metadata_file = 'E:/everything/datasets/song_data.csv'

'''
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
'''

song_df_1 = pandas.read_table(triplets_file, header=None)
# df_1 = pandas.DataFrame(song_df_1, columns=['user_id', 'song_id', 'listen_count'])
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

song_df_2 = pandas.read_csv(songs_metadata_file)
song_df_3 = song_df_2

song_df = song_df_1.merge(song_df_3, on='song_id', how='left')
song_df.head()
song_grouped = song_df.groupby(['song_id']).agg(dict(listen_count='count')).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song_id'],ascending=[0,1])

users = song_df['user_id'].unique()
print(len(users))

songs = song_df['song_id'].unique()
print(len(songs))

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state = 0)

pm = RecommenderSystem.popularity_recommender_py()
pm.create(train_data,'user_id','title')
user_id = users[5]
pm.recommend(user_id)

is_model = RecommenderSystem.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'title')

# user_id = users[5]
user_items = is_model.get_user_items(user_id)
print("--------------------------------------------------------------------------------------")
print("TRAINING DATA SONGS FOR THE USER USER_ID :\n%s : "%user_id)
print("--------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("--------------------------------------------------------------------------------")
print("RECOMMEND SONGS FOR THE USER ....")
print("--------------------------------------------------------------------------------")

is_model.recommend(user_id)

# is_model.get_similar_items(["U Smile - Justin Bieber"])
# print("--------------------------------------------------------------------------------")
# print("RECOMMENDATION PROCESS BASED ON ITEMSET ....")
# print("--------------------------------------------------------------------------------")

# for i in similiar_item:
#     print(i)

# is_model.recommend(["U Smile - Justin Bieber"])

