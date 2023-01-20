import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# task 1
frame_t = pd.read_csv('titles.csv')
fr = pd.read_csv('titles.csv', index_col='type')
movies = fr.loc['MOVIE', 'imdb_score']
shows = fr.loc['SHOW', 'imdb_score']

plt.subplot(1, 2, 1)

plt.xticks(np.arange(0, 10, 0.2), rotation='vertical')
plt.hist(movies, 50)  # step should be 0.2 -> 10/0.2 = 50
plt.xlabel("IMDB score")
plt.ylabel("Number of movies")
plt.subplot(1, 2, 2)

plt.xticks(np.arange(0, 10, 0.2), rotation='vertical')
plt.hist(shows, 50)
plt.xlabel("IMDB score")
plt.ylabel("Number of shows")
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


# task 3
year = frame_t[['release_year', 'imdb_score']]

year_d = {}
for ind in year.index:
    if int(year['release_year'][ind]) > 1999:
        if year['release_year'][ind] not in year_d.keys():
            year_d[year['release_year'][ind]] = [[], []]

        year_d[year['release_year'][ind]][0].append(year['imdb_score'][ind])
        if float(year['imdb_score'][ind]) >= 8.0:
            year_d[year['release_year'][ind]][1].append(year['imdb_score'][ind])

av_year = {}
for key, value in year_d.items():
    av_year[key] = (len(value[1]) / len(value[0])) * 100

highest_percent = max(av_year.values())
best_year = max(av_year, key=av_year.get)
print(f'the best year was {best_year} with {int(highest_percent)}% of successful projects')


plt.bar(av_year.keys(), av_year.values())
plt.xlabel('years')
plt.ylabel('percent of successful shows/movies')
plt.show()


