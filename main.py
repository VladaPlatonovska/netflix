import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# task 1
frame_t = pd.read_csv('titles.csv')
fr = pd.read_csv('titles.csv', index_col='type')
movies = fr.loc['MOVIE', 'imdb_score']
shows = fr.loc['SHOW', 'imdb_score']

plt.subplot(1, 2, 1)
plt.hist(movies, 50)  # step should be 0.2 -> 10/0.2 = 50
plt.subplot(1, 2, 2)
plt.hist(shows, 50)
plt.show()


# task 2
age = frame_t.loc[:, 'age_certification']
age_r = {}
for r in age:
    if r not in age_r.keys():
        age_r[r] = 0
    age_r[r] += 1
plt.pie(age_r.values(), labels=age_r.keys())
plt.show()
