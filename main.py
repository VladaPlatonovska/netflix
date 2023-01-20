import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# task 1
frame_t = pd.read_csv('titles.csv')
fr = pd.read_csv('titles.csv', index_col='type')
movies = fr.loc['MOVIE', 'imdb_score']
shows = fr.loc['SHOW', 'imdb_score']

plt.figure(figsize=(12, 8))
plt.rc('xtick', labelsize=5)
plt.subplot(1, 2, 1)
plt.xticks(np.arange(0, 10, 0.2), rotation='vertical')
plt.hist(movies, 50)  # step should be 0.2 -> 10/0.2 = 50
plt.xlabel("IMDB score")
plt.ylabel("Number of movies")
plt.xticks(np.arange(0, 10, 0.2), rotation='vertical')
plt.axvline(movies.mean(), color='sienna', linestyle='dashed', linewidth=1)


plt.subplot(1, 2, 2)
plt.rc('xtick', labelsize=5)
plt.xticks(np.arange(0, 10, 0.2), rotation='vertical')
plt.hist(shows, 50)
plt.xlabel("IMDB score")
plt.ylabel("Number of shows")
plt.axvline(shows.mean(), color='sienna', linestyle='dashed', linewidth=1)
plt.show()


# task 2
age = frame_t.loc[:, 'age_certification']
age_r = {}
for r in age:
    if r not in age_r.keys():
        age_r[r] = 0
    age_r[r] += 1
plt.rc('xtick', labelsize=10)
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
plt.axhline(highest_percent, color='sienna', linestyle='dashed', linewidth=1)

plt.rc('xtick', labelsize=10)
plt.bar(av_year.keys(), av_year.values())
plt.xlabel('years')
plt.ylabel('percent of successful shows/movies')
plt.show()


# task 5
top_m = frame_t[['id', 'genres', 'imdb_score']]
sorted_top = top_m.sort_values(by='imdb_score', ascending=False)
top_1000 = sorted_top[:1000]

dict_genres = {}
for t in top_1000['genres']:
    g = t.split(' ')

    for gen in g:
        if len(gen) > 3:
            if gen[0] == '[':
                gen = gen[2:]
            elif gen[0] == "'":
                gen = gen[1:]

            if gen[-1] == ']' or gen[-1] == ',':
                gen = gen[:-2]
            elif gen[-1] == "'":
                gen = gen[:-1]
        if gen == '[]':
            gen = 'none'
        if gen not in dict_genres.keys():
            dict_genres[gen] = 0

        dict_genres[gen] += 1


ax = plt.barh(list(dict_genres.keys()), dict_genres.values())
plt.xlabel('number of shows/movies')
plt.ylabel('genres')
plt.tight_layout()
for i in ax.patches:
    plt.text(i.get_width(), i.get_y() + 0.2,
             str(round((i.get_width()), 2)),
             fontsize=8, color='black', ha='left', va='baseline')
plt.show()

# plt.text(x, y, text, other things)
