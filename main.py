import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# frame_t = pd.read_csv('titles.csv')
fr = pd.read_csv('titles.csv', index_col='type')
# dict_m = {}
# a = frame_t[['type', 'imdb_score']]
# # print(a)
# m_type = frame_t.loc[:, 'type']
# imdb_score = frame_t.loc[:, 'imdb_score']
# for t in m_type:
#     if t not in dict_m.keys():
#         dict_m[t] = []
# imdb_movies = []
# # print(dict_m)

movies = fr.loc['MOVIE', 'imdb_score']
shows = fr.loc['SHOW', 'imdb_score']
plt.xticks(np.arange(0, 11, 0.2))
# plt.step(10, 100)
plt.subplot(1, 2, 1)
plt.hist(movies)
plt.subplot(1, 2, 2)
plt.hist(shows)
plt.show()
# print(movies)
# print(imdb_movies)


# print(imdb_score)
# plt.hist(imdb_score)
# plt.axis([0, 10, 0, 120])
# plt.show()

