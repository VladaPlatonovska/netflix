import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


titles = pd.read_csv('titles.csv')
credits = pd.read_csv('credits.csv')

# task 1
movies = titles[titles['type'] == 'MOVIE']['imdb_score'].dropna()
shows = titles[titles['type'] == 'SHOW']['imdb_score'].dropna()

m_mean = movies.mean()
s_mean = shows.mean()
if m_mean > s_mean:
    print("movies' average mark is greater")
else:
    print("shows' average mark is greater")

plt.figure(figsize=(10, 6))
plt.axvline(m_mean, color='sienna', linestyle='dashed', linewidth=1)

plt.rc('xtick', labelsize=5)
plt.xticks(np.arange(0, 10, 0.2), rotation='vertical')
plt.xlabel("IMDB score")
plt.ylabel("Number of shows/movies")
plt.axvline(s_mean, color='blue', linestyle='dashed', linewidth=1)

sns.histplot(titles, x='imdb_score', bins=np.arange(0, 10, 0.2), hue='type')
plt.show()


# task 2

values = titles[titles['type'] == 'SHOW']['age_certification'].dropna()
labels, values = np.unique(values, return_counts=True)
plt.pie(values, labels=labels)
plt.show()


# task 3
year = titles[['release_year', 'imdb_score']]

years = titles[titles['release_year'] > 1999]
y = years.groupby(by='release_year').co
print(y)



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

plt.rc('xtick', labelsize=10)
bar_list = plt.bar(av_year.keys(), av_year.values())
bar_list[int(best_year) - 2000].set_facecolor('red')
plt.xlabel('years')
plt.ylabel('percent of successful shows/movies')
plt.show()


# task 4

top_1000 = titles.sort_values(by='imdb_score', ascending=False).head(1000)
merged_d = pd.merge(top_1000, credits[credits['role'] == 'ACTOR'], how='inner', on=['id'])
top10_act = merged_d.groupby(by='name')['id'].count().sort_values(ascending=False).head(10)
print(top10_act)


# task 5
top_m = titles[['id', 'genres', 'imdb_score']]
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
