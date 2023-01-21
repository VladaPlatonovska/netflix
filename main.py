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
years = titles[titles['release_year'] > 1999]
y = years[years['imdb_score'] >= 8.0].dropna()
years_grouped = years.groupby(by='release_year')['release_year'].count().sort_index()
years_g_score = y.groupby(by='release_year')['imdb_score'].count().sort_index()
s = pd.DataFrame({'lab': years_grouped.keys(), 'val': ((years_g_score / years_grouped)*100)})

bar_list = plt.bar(x=s['lab'], height=s['val'])
bar_list[int(s["lab"][s["val"] == max(s["val"])]) - 2000].set_facecolor('red')
plt.show()


# task 4

top_1000 = titles.sort_values(by='imdb_score', ascending=False).head(1000)
merged_d = pd.merge(top_1000, credits[credits['role'] == 'ACTOR'], how='inner', on=['id'])
top10_act = merged_d.groupby(by='name')['id'].count().sort_values(ascending=False).head(10)
print(f'TOP-10 ACTORS\n{top10_act.to_string(header=False)}')


# task 5
top_1000 = titles.sort_values(by='imdb_score', ascending=False).head(1000)
genres_d = {}

for t in top_1000['genres']:
    no_comma = t.replace(',', '')
    no_marks = no_comma.replace("'", '')
    no_par = no_marks.strip("[]").split(' ')
    for genre in no_par:
        if genre == '':
            genre = 'none'
        if genre not in genres_d.keys():
            genres_d[genre] = 0

        genres_d[genre] += 1

data_genres = pd.DataFrame(({'genres': genres_d.keys(), 'number_of_pr': genres_d.values()}))

ax = plt.barh(data_genres['genres'], data_genres['number_of_pr'])
plt.xlabel('number of shows/movies')
plt.ylabel('genres')
plt.tight_layout()
for i in ax.patches:
    plt.text(i.get_width(), i.get_y() + 0.2,
             str(round((i.get_width()), 2)),
             fontsize=8, color='black', ha='left', va='baseline')
plt.show()