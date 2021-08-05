#!/usr/bin/env python
# coding: utf-8

# # Сборный проект " Компьютерные игры"

# ### Содержание
# 1. [Цель и описание работы](#aim)
# 2. [Анализ и предобработка данных](#pre_work)
# 3. [Исследовательский анализ данных](#data_analysis)
# 4. [Портрет пользователя каждого региона](#user_portrait)
# 5. [Проверка гипотез](#hypo_check)
# 6. [Общий вывод](#conclusion)

# Импорт основных библиотек

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as st


# Функции, используемые в проекте

# In[2]:


def Get_Median(df_med,platform,genre,year,med_field):
    ''' получает медианное значение по году, жанру и платформе из df_med для заданного поля 
    '''
    median=0
    for rows in df_med.iterrrows():
        if (rows.platform==platform) & (rows.genre==genre) & (rows.year_of_release==year):
            median=rows.med_field
    return median

def Fill_Year(df,year_df):
    ''' заполняем пустые значения в поле год
    '''
    for i,rows in df.iterrows():
        if rows['year_of_release']==0:
            genre=rows['genre']
            platform=rows['platform']
            year=Get_Year(year_df, genre, platform)
            df.at[i,'year_of_release']=year
            
            
def Get_Year(df_years, genre, platform):
    ''' получаем значение года из df_years для заданного жанра и заданной платформы
    '''
    year=0
    for i,rows in df_years.iterrows():
        if (rows['genre']==genre) & (rows['platform']==platform):
            year=rows['year_of_release']
    return year     


# ## 1. Цель и описание работы  <a id= aim></a>
# Интернет-магазин «Стримчик» продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании. Имеются данные до 2016 года и мы планируем кампанию на 2017-й. 
# В наборе данных попадается аббревиатура ESRB (Entertainment Software Rating Board) — это ассоциация, определяющая возрастной рейтинг компьютерных игр. ESRB оценивает игровой контент и присваивает ему подходящую возрастную категорию, например, «Для взрослых», «Для детей младшего возраста» или «Для подростков».

# ## 2.Анализ и предобработка данных <a id= pre_work></a>

# Откроем и изучим файл с данными.

# В файле содержится следующая информация:
# 
# •	Name — название игры
# 
# •	Platform — платформа
# 
# •	Year_of_Release — год выпуска
# 
# •	Genre — жанр игры
# 
# •	NA_sales — продажи в Северной Америке (миллионы проданных копий)
# 
# •	EU_sales — продажи в Европе (миллионы проданных копий)
# 
# •	JP_sales — продажи в Японии (миллионы проданных копий)
# 
# •	Other_sales — продажи в других странах (миллионы проданных копий)
# 
# •	Critic_Score — оценка критиков (максимум 100)
# 
# •	User_Score — оценка пользователей (максимум 10)
# 
# •	Rating — рейтинг от организации ESRB (англ. Entertainment Software Rating Board). Эта ассоциация определяет рейтинг 
# компьютерных игр и присваивает им подходящую возрастную категорию.

# In[3]:


games=pd.read_csv('/datasets/games.csv', sep=',')


# In[4]:


games.info()
games.head(5)


# В файле содержится 16715 строк, есть пропуски в полях с названием игры, годом выхода, жанром, оценкой критиков, оценкой пользователей и рейтингом.

# Заменим названия столбцов(приведем все названия к нижнему регистру):
# 

# In[5]:


games.columns=games.columns.str.lower()


# In[6]:


games.tail(5)


# Преобразуем данные в стобце с годом к целому типу, год не может быть числом с запятой:

# In[7]:


games['year_of_release']= games['year_of_release'].fillna(0)
games['year_of_release']= games['year_of_release'].astype('int')


# В столбце user_score есть пропуски и есть строчное значение tbd. Для дальнейшей работы мне нужно привести данные к числовому типу, поэтому все NaN заменим на -10(чтобы не перепутать с играми с 0й оценкой), а строчное значение tbd на -100. 

# In[8]:


games['user_score']= games['user_score'].fillna(-10)
games['critic_score']=games['critic_score'].fillna(-10)

for i,rows in games.iterrows():
    if rows['user_score']=="tbd":
        games.at[i,'user_score']=-100
    

games['user_score']=games['user_score'].astype('float')

games['rating']=games['rating'].fillna(0)

games.info()


# ### Обработка пропусков

# Проверим в каких строках пропущены названия игр:

# In[9]:


games_na_names=games[games['name'].isna()==True]


# Видно, что у 2х игр есть только название платформы и продажи. Года выпуска и жанра нет, соответсвенно для прогнозов или анализа данные бесполезны. Данные строки будут удалены:

# In[10]:


games_na_names


# In[11]:


games=games.drop(games[games.name.isna()==True].index)


# В поле year_of_release есть пропущенные значения. Проверим сколько и  у каких жанров и платформ пропущено:

# In[12]:


tmp_na_year=games[games['year_of_release']==0]


# In[13]:


tmp_na_year_by_genre=tmp_na_year.pivot_table(index=['genre','platform'], values='name', aggfunc='count').sort_values(by='name', ascending=False)


# In[14]:


tmp_na_year_by_genre


# Посмотрим в какие года выходило больше всего игр каждого жанра для каждой платформы.

# In[15]:


tmp_year_stat=games[games['year_of_release']>0].pivot_table(index=['genre','platform','year_of_release'], values='name', aggfunc='count').sort_values(by='name', ascending=False).reset_index()


# In[16]:


tmp_year_stat


# Избавимся от дубликатов, таким образом у нас останутся только года, когда выходило больше всего игр для данной платформы данного жанра:

# In[17]:


tmp_year_stat=tmp_year_stat.drop_duplicates(['genre','platform']).sort_index()


# Заполним пропуски в столбце с годом выхода значениями годов, когда выходило больше всего игр

# In[18]:


Fill_Year(games,tmp_year_stat)


# In[19]:


games.info()


# Попробуем проверить есть ли зависимость между продажами и оценкой критиков, продажами и оценкой пользователей или оценками критиков и оценкой пользователей. 

# Добавим столбец sales_total, где будет указана итоговая сумма продаж:

# In[20]:


games['sales_total']=games.na_sales+games.eu_sales+games.jp_sales+games.other_sales


# In[21]:


tmp_games=games[(games['critic_score']>0) & (games['user_score']>0)]
tmp_games


# Попробуем выявить корреляции между продажами и оценками критиков:

# In[22]:


corrmat = tmp_games.corr()
f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# In[23]:


round(corrmat,2)


# Сильной зависимости между продажами и оценками нет. Также нет сильной связи между оценками критиков и оценками пользователей.

# В данном случае не стоит заполнять данные по оценкам медианными значениями, это может сильно исказить реальность. Если бы была сильная зависимость между продажами и рейтингом, возможно дополнительные вычисления медианы позволили бы дополнить данные более или менее точными значениями. Поэтому пропуски в оценках будут заполнены маркерными значениями:
# 
# -tbd: оценка не выставлялась или слишком мала, промаркируем отрицательным значением -100
# 
# -NaN: нет данных, промаркируем отрицательным значением -10
# 

# In[24]:


games.info()


# #### Вывод: 
# На данном этапе была сделана предобработка данных и заполнены пропуска, где возможно. Выявить закономерности в количестиве пропуском не удалось, поэтому предполагаю, что здесь данные отсутствуют в зависимости от неизвестных факторов. Выявить сильную связь между оценками и другими параметрами не удалось, поэтому было принято решение промаркировать пропуска отрицательными значениями для дальнейшей фильтрации. Также было сделано преобразование типов в столбцах с оценками к числовым значениям.

# ## 3. Исследовательский анализ данных<a id= data_analysis></a>

# Посмотрим сколько игр выпускалось в разные годы.

# In[25]:


games_by_year=games.pivot_table(index='year_of_release', values='name',aggfunc='count').reset_index()


# In[26]:


games_by_year


# In[27]:


fig=go.Figure()

fig.add_trace(go.Scatter(x=games_by_year['year_of_release'],y=games_by_year['name']))
fig.update_layout(title="Динамика выпуска игр по годам",
                  xaxis_title="Год",
                  yaxis_title="Количество игр")

fig.show()


# Видно, что активный рост в игровой индустрии наблюдался с 2000 по 2009 года, дальше наблюдается спад. В данном случае интересен период с 2004 года, когда наблюдался постоянный рост от года к году и далее.

# Посмотрим, как менялись продажи по платформам. Выберем платформы с наибольшими продажами и посмотрим, как менялись продажи по годам

# In[28]:


tmp_best_platforms=games.pivot_table(index='platform',values='sales_total',aggfunc='sum').reset_index().sort_values(by='sales_total',ascending=False)


# In[29]:


tmp_best_platforms


# Возьмем 1-е 6 платформ, далее итоговые продажи падают более, чем в 2 раза, поэтому рассматривать их не будем.

# In[30]:


best_names=tmp_best_platforms[:6]


# In[31]:


best_names


# Сформируем 6 срезов с данными для визуализации:

# In[32]:


PS2_by_year=games[games['platform']=='PS2'].pivot_table(index='year_of_release',values='sales_total',aggfunc='sum').reset_index()
X360_by_year=games[games['platform']=='X360'].pivot_table(index='year_of_release',values='sales_total',aggfunc='sum').reset_index()
PS3_by_year=games[games['platform']=='PS3'].pivot_table(index='year_of_release',values='sales_total',aggfunc='sum').reset_index() 
Wii_by_year=games[games['platform']=='Wii'].pivot_table(index='year_of_release',values='sales_total',aggfunc='sum').reset_index() 
DS_by_year=games[games['platform']=='DS'].pivot_table(index='year_of_release',values='sales_total',aggfunc='sum').reset_index()
PS_by_year=games[games['platform']=='PS'].pivot_table(index='year_of_release',values='sales_total',aggfunc='sum').reset_index()


# In[33]:


fig=go.Figure()


fig.add_trace(go.Scatter(x=PS2_by_year['year_of_release'],y=PS2_by_year['sales_total'], name="PS2"))
fig.add_trace(go.Scatter(x=X360_by_year['year_of_release'],y=X360_by_year['sales_total'],name="X360"))
fig.add_trace(go.Scatter(x=PS3_by_year['year_of_release'],y=PS3_by_year['sales_total'],name="PS3"))
fig.add_trace(go.Scatter(x=Wii_by_year['year_of_release'],y=Wii_by_year['sales_total'],name="Wii"))
fig.add_trace(go.Scatter(x=DS_by_year['year_of_release'],y=DS_by_year['sales_total'],name="DS"))
fig.add_trace(go.Scatter(x=PS_by_year['year_of_release'],y=PS_by_year['sales_total'],name="PS"))

fig.update_layout(title="Динамика продаж игр по годам",
                  xaxis_title="Год",
                  yaxis_title="Продажи, млн. копий")

fig.show()


# Заметно, что средний срок жизни платформы 10 лет, на графиках по 6 ти лучшим видно, что с появления до исчезновения проходит от 9 до 11 лет.

# Согласно данным Википедии последнее поколение консолей выпускается с 2012 года и по текущее время и скорее всего рынок игр будет ориентирваься именно на них и новые поколения, поэтому 2012-2016 года будут для нас актуальным периодом. Консоли прошлых поколений(до 2012 и старше) к 2017 году уже не интересны.

# In[34]:


actual_consoles=games[(games['year_of_release']>=2012) & (games['year_of_release']<2016)].pivot_table(index=['platform'],columns='year_of_release', values='sales_total', aggfunc='sum')
actual_consoles


# Данные за 2016 год могут быть неполными по условию задачи, поэтому проследим тренды до 2015 года включительно.

# <div style="background: #cceeaa; padding: 5px; border: 1px solid green; border-radius: 5px;">
#     <font color='green'> <b><u>КОММЕНТАРИЙ РЕВЬЮЕРА</u></b>
# </font>
# <font color='green'><br>да. принято)

# In[35]:


actual_consoles.plot(kind='bar')


# Пока в лидерах 2 платформы: XOne и PS4, у остальных наблюдается падение в продажах.

# Создадим 2 DataFrame c данными о продажах XOne и PS4 по годам.

# In[36]:


potential_platforms=games[(games['platform']=="XOne") |(games['platform']=="PS4") ]


# In[37]:


potential_platforms


# In[38]:


XOne=games[(games['platform']=="XOne")]
PS4=games[(games['platform']=="PS4")]
DS3=games[(games['platform']=="3DS")]
DS=games[(games['platform']=="DS")]
PC=games[(games['platform']=="PC")]
PS3=games[(games['platform']=="PS3")]
PSP=games[(games['platform']=="PSP")]
PSV=games[(games['platform']=="PSV")]
Wii=games[(games['platform']=="Wii")]
WiiU=games[(games['platform']=="WiiU")]
X360=games[(games['platform']=="X360")]



fig = go.Figure()
fig.add_trace(go.Box(y=XOne['sales_total'],name="XOne"))
fig.add_trace(go.Box(y=PS4['sales_total'],name="PS4"))
fig.add_trace(go.Box(y=DS3['sales_total'],name="3DS"))
fig.add_trace(go.Box(y=DS['sales_total'],name="DS"))
fig.add_trace(go.Box(y=PC['sales_total'],name="PC"))
fig.add_trace(go.Box(y=PS3['sales_total'],name="PS3"))
fig.add_trace(go.Box(y=PSP['sales_total'],name="PSP"))
fig.add_trace(go.Box(y=PSV['sales_total'],name="PSV"))
fig.add_trace(go.Box(y=Wii['sales_total'],name="Wii"))
fig.add_trace(go.Box(y=WiiU['sales_total'],name="WiiU"))
fig.add_trace(go.Box(y=X360['sales_total'],name="X360"))



fig.update_layout(
    title='Потенциальные платформы',
    yaxis=dict(range=[0,2.5],
        autorange=False,
        showgrid=True
    ))

fig.show()


# Видно, что у всех платформ после 1.5 млн копий лежат выбросы. Попробую убрать их и посмотреть, что получится:

# In[39]:


games_upd=games[games['sales_total']<=1.5]

XOne=games_upd[(games_upd['platform']=="XOne")]
PS4=games_upd[(games_upd['platform']=="PS4")]
DS3=games_upd[(games_upd['platform']=="3DS")]
DS=games_upd[(games_upd['platform']=="DS")]
PC=games_upd[(games_upd['platform']=="PC")]
PS3=games_upd[(games_upd['platform']=="PS3")]
PSP=games_upd[(games_upd['platform']=="PSP")]
PSV=games_upd[(games_upd['platform']=="PSV")]
Wii=games_upd[(games_upd['platform']=="Wii")]
WiiU=games_upd[(games_upd['platform']=="WiiU")]
X360=games_upd[(games_upd['platform']=="X360")]



fig = go.Figure()
fig.add_trace(go.Box(y=XOne['sales_total'],name="XOne"))
fig.add_trace(go.Box(y=PS4['sales_total'],name="PS4"))
fig.add_trace(go.Box(y=DS3['sales_total'],name="3DS"))
fig.add_trace(go.Box(y=DS['sales_total'],name="DS"))
fig.add_trace(go.Box(y=PC['sales_total'],name="PC"))
fig.add_trace(go.Box(y=PS3['sales_total'],name="PS3"))
fig.add_trace(go.Box(y=PSP['sales_total'],name="PSP"))
fig.add_trace(go.Box(y=PSV['sales_total'],name="PSV"))
fig.add_trace(go.Box(y=Wii['sales_total'],name="Wii"))
fig.add_trace(go.Box(y=WiiU['sales_total'],name="WiiU"))
fig.add_trace(go.Box(y=X360['sales_total'],name="X360"))



fig.update_layout(
    title='Потенциальные платформы',
    yaxis=dict(range=[0,0.6],
        autorange=False,
        showgrid=True
    ))

fig.show()


# .
# 
# 

# Рассмотрим более подробно платформу PS4. Уберем выбросы и проверим есть ли связь между отзывами критиков, отзывами пользователей и продажами.

# In[40]:


PS4=PS4[(PS4['sales_total']<=1.74) & (PS4['user_score']>=0)]


# In[41]:


fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x = PS4['sales_total'], y = PS4['user_score'])
plt.xlabel("user_score")
plt.ylabel("sales_total")
plt.show()


# По диаграмме видно, что линейной зависимости между продажами и оценками пользователей нет. Проверим, есть ли зависимость между оценками критиков и продажами:

# In[42]:


PS4=PS4[(PS4['sales_total']<=1.74) & (PS4['critic_score']>=0)]
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x = PS4['sales_total'], y = PS4['critic_score'])
plt.xlabel("critic_score")
plt.ylabel("sales_total")
plt.show()


# Аналогично, нельзя выявить зависимость между продажами и оценками критиков.

# In[43]:


print(f'Корреляция между продажами и оценками пользователей:{round(PS4["sales_total"].corr(PS4["user_score"]),2)}')


# In[44]:


print(f'Корреляция между продажами и оценками критиков:{round(PS4["sales_total"].corr(PS4["critic_score"]),2)}')


# Коэффициенты подтверждают отсутствие связи между оценками и продажами для платформы PS4. Это лишний раз доказывает, что не всегда можно полагаться на рейтинги пользователей и критиков при выборе игр и скорее всего при покупке игры решающими факторами ни оценка критиков, ни рейтинг пользователей не становятся. Аналогичная ситуация наблюдается и в целом по остальным играм. В предыдущем пункте при обработке пропусков было выявлено, что нет зависимости между оценками и продажами.

# ![image.png](attachment:image.png)

# Посмотрим на общее распределение игр по жанрам.

# In[45]:


genre_split=games.pivot_table(index='genre',values='sales_total', aggfunc=['sum','count','mean'])
genre_split.columns=['sum','count','mean']
round(genre_split.sort_values(by='mean', ascending=False),2)


# Видно, что самыми популярными являются игры Action и Sports жанров, а аутсайдерами Puzzle, Adventure и Strategy жанры.

# ### Вывод:
# Время жизни платформы от 9 до 11 лет, дальше консоль отмирает. Если смотреть данные за актуальный период, то сейчас наибольшим потенциалом обладают консоли XOne и PS4. В качестве актуального периода взяты данные с 2012 по 2015й года, в 2012 начался выпуск последнего поколения консолей, а 2016 год был исключен из-за того, что данные неполные. Самыми прибыльными жанрами являются Action и Sport, однако, если судить по средней продаже, то Platform, Shooter и Role-Playing приносят больше.

# ## 4.Портрет пользователя каждого региона<a id= user_portrait></a>

# Выделим топ-5 платформ для пользователей каждого из регионов NA, EU, JP.

# ### Северная Америка

# Посмотрим какие платфоры являются самыми популярными:

# In[46]:


NA_top5_platforms=games.pivot_table(index='platform',values='na_sales',aggfunc='sum').reset_index().sort_values(by='na_sales',ascending=False)[:5]


# In[47]:


plt.figure(figsize=(16,8))

ax1 = plt.subplot(121, aspect='equal')
NA_top5_platforms.plot(kind='pie', y = 'na_sales', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, labels=NA_top5_platforms['platform'], legend = True)


# Посмотрим какие жанры популярны в Северной Америке:

# In[48]:


NA_top5_genres=games.pivot_table(index='genre',values='na_sales',aggfunc='sum').reset_index().sort_values(by='na_sales',ascending=False)[:5]


# In[49]:


plt.figure(figsize=(16,8))
ax1 = plt.subplot(121, aspect='equal')
NA_top5_genres.plot(kind='pie', y = 'na_sales', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, labels=NA_top5_genres['genre'], legend = True)


# Геймеры в Северной Америке предпочитают следующие консоли:
# 
# 1. X360
# 2. PS2
# 3. Wii
# 4. PS3
# 5. DS
# 
# Самыми популярными игровыми жанрами являются:
# 
# 1. Action
# 2. Sports
# 3. Shooter
# 4. platform
# 5. Misc
# 

# ### Япония

# Посмотрим какие платфоры являются самыми популярными в Японии:

# In[50]:


JP_top5_platforms=games.pivot_table(index='platform',values='jp_sales',aggfunc='sum').reset_index().sort_values(by='jp_sales',ascending=False)[:5]


# In[51]:


plt.figure(figsize=(16,8))
# plot chart
ax1 = plt.subplot(121, aspect='equal')
JP_top5_platforms.plot(kind='pie', y = 'jp_sales', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, labels=JP_top5_platforms['platform'], legend = True)


# Распределение предпочтений по жанрам:

# In[52]:


JP_top5_genres=games.pivot_table(index='genre',values='jp_sales',aggfunc='sum').reset_index().sort_values(by='jp_sales',ascending=False)[:5]
plt.figure(figsize=(16,8))
ax1 = plt.subplot(121, aspect='equal')
JP_top5_genres.plot(kind='pie', y = 'jp_sales', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, labels=JP_top5_genres['genre'], legend = True)


# В Японии предпочитают следующие консоли:
# 
# 1. DS
# 2. PS
# 3. PS2
# 4. SNES
# 5. 3DS
# 
# Что касается жанров, то здесь предпочтения следующие:
# 
# 1. Role Playing ( доля по сранению с остальными выше в 2 и более раз)
# 2. Action
# 3. Sports
# 4. Platform
# 5. Misc

# ### Европа

# Посмотрим какие предпочтения у европейских пользователей.

# In[53]:


EU_top5_platforms=games.pivot_table(index='platform',values='eu_sales',aggfunc='sum').reset_index().sort_values(by='eu_sales',ascending=False)[:5]
plt.figure(figsize=(16,8))
ax1 = plt.subplot(121, aspect='equal')
EU_top5_platforms.plot(kind='pie', y = 'eu_sales', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, labels=EU_top5_platforms['platform'], legend = True)


# Посмотрим какие жанры предпочитают в Европе:

# In[54]:


EU_top5_genres=games.pivot_table(index='genre',values='eu_sales',aggfunc='sum').reset_index().sort_values(by='eu_sales',ascending=False)[:5]
plt.figure(figsize=(16,8))
ax1 = plt.subplot(121, aspect='equal')
EU_top5_genres.plot(kind='pie', y = 'eu_sales', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, labels=EU_top5_genres['genre'], legend = True)


# В Европе самыми популярными являются следующие консоли:
# 
# 1. PS2
# 2. PS3
# 3. X360
# 4. Wii
# 5. PS

# Наиболее популярными являются следующие жанры:
# 
# 1. Action
# 2. Sports
# 3. Shooter
# 4. Racing
# 5. Misc

# Посмотрим как рейтинг ESRB влияет на продажи в каждом регионе.

# In[55]:


ESRB_data=games[games['rating']!=0].pivot_table(index='rating',values=['eu_sales','na_sales','jp_sales'],aggfunc='sum').reset_index()
ESRB_data


# In[56]:


fig=go.Figure()


fig.add_trace(go.Scatter(x=ESRB_data['rating'],y=ESRB_data['eu_sales'], name="eu_sales"))
fig.add_trace(go.Scatter(x=ESRB_data['rating'],y=ESRB_data['jp_sales'], name="jp_sales"))
fig.add_trace(go.Scatter(x=ESRB_data['rating'],y=ESRB_data['na_sales'], name="na_sales"))

fig.update_layout(title="Продажи и рейтинг ESRB",
                  xaxis_title="ESRB рейтинг",
                  yaxis_title="Продажи, млн. копий")

fig.show()


# Видно, что в 3х регионах одинаковый тренд продаж и рейтинга: самыми популярными являются игры:
# 1. E - для всех
# 2. Игры категории M- для взрослых
# 3. Игры категории T - для подростков
# 

# ### Вывод: 
# Видно, как отличаются предпочтения у европейский и американских пользователей от предпочтений японских пользователей. В Европе и Америке наибольшей популярностью пользуются PS2, PS3 и X360, то в Японии большую долю занимают DS и PS. В жанрах аналогично предпочтения меняются в зависимости от рынков: в Японии предочитают Role Playing, а в Европе Sport и Action. Оценка ESRB также влияет на продажи игр, причем тренд одинаковый независимо от рынка: самыми популярными являются игры для всех(Е), затем игры с маркировокой для взрослых и для подростков.

# ## 5. Проверка гипотез<a id= hypo_check></a>

# Проверим гипотезу, что средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
# 
# Нулевая гипотеза H0: средние пользовательские рейтинги платформ Xbox One и PS одинаковые.
# 
# Альтернативная гипотеза: средние пользовательские рейтинги платформ Xbox One и PS разные
# 

# In[57]:


XBox1=games[(games['platform']=='XOne') & (games['user_score']>0)]
PS=games[(games['platform']=='PS') & (games['user_score']>0)]


# Проверим размеры наших данных:

# In[58]:


len(XBox1)


# In[59]:


len(PS)


# Сформируем 2 выборки по 100 строк.

# In[60]:


XBox1=XBox1.sample(100)
PS=PS.sample(100)
                   


# Зададим параметр alpha равным 0,05

# In[61]:


alpha=0.05


# Вычислим дисперсию:

# In[62]:


round(XBox1['user_score'].var(ddof=1),2)


# In[63]:


round(PS['user_score'].var(ddof=1),2)


# Проверим дисперсии выборок с помощью теста Левена:

# In[64]:


var_levene =st.levene(XBox1['user_score'],PS['user_score'], center='median')
var_levene


# По результатам теста Левена  p-value больше критического значения 0.05, т.е. гипотезу о равенстве дисперсий можно отвергнуть. Построим box plot для наших выборок, чтобы еще раз проверить:

# In[65]:


fig = go.Figure()
fig.add_trace(go.Box(y=XBox1['user_score'],name="XBox", fillcolor="yellow"))
fig.add_trace(go.Box(y=PS['user_score'],name="PS", fillcolor="violet"))



fig.update_layout(
    title='Пользовательские оценки XBox и PS4: статистические показатели',
    yaxis=dict(
        autorange=True,
        showgrid=True
    ))

fig.show()


# Дисперсии наших выборок не равны между собой. Для проверки таких гипотез применяется двухсторонний tтест.

# In[66]:


results = st.ttest_ind(XBox1['user_score'], PS['user_score'], equal_var=False)
if results.pvalue < alpha:
    print("Отвергаем нулевую гипотезу")
else:
    print("Отвергать нулевую гипотезу нельзя")


# Гипотеза о том, что средние пользовательские оценки для двух консолей XBox1 и PS одинаковые не подвердилась.

# Проверим гипотезу о том, что пользовательские рейтинги жанров Action и Sports разные.
# Нулевая гипотеза Н0: пользовательские рейтинги жанров Action и Sport одинаковые.
# Альтернативная гипотеза H1: пользовательские рейтинги жанров Action и Sport отличаются.

# In[67]:


Action=games[(games['genre']=='Action') & (games['user_score']>0)]
Sport=games[(games['genre']=='Sports') & (games['user_score']>0)]


# Вычислим объемы наших совокупностей:

# In[68]:


len(Action)


# In[69]:


len(Sport)


# Возьмем для исследования выборки по 250 элементов:

# In[70]:


Action=Action.sample(250)
Sport=Sport.sample(250)


# Вычислим дисперсию каждой выборки:

# In[71]:


round(Action['user_score'].var(ddof=1),2)


# In[72]:


round(Sport['user_score'].var(ddof=1),2)


# Проверим выборки с помощью теста Левена:

# In[73]:


st.levene(Action['user_score'],Sport['user_score'], center='mean')


# p-value меньше 0,05, значит выборки однородные и различие между дисперсиями не значимо.

# In[74]:


fig = go.Figure()
fig.add_trace(go.Box(y=Action['user_score'],name="Action"))
fig.add_trace(go.Box(y=Sport['user_score'],name="PS"))



fig.update_layout(
    title='Пользовательские оценки жанров Action и Sport:',
    yaxis=dict(
        autorange=True,
        showgrid=True
    ))

fig.show()


# In[75]:


results = st.ttest_ind(Sport['user_score'], Action['user_score'], equal_var=True)
if results.pvalue < alpha:
    print("Отвергаем нулевую гипотезу")
else:
    print("Отвергать нулевую гипотезу нельзя")


# Отвергать нулевую гипотезу о том, что средние пользовательские оценки жанров Sport и Action нельзя.

# ### Вывод:  
# Были проверены 2 гипотезы: средняя пользовательская оценка двух консолей XBox1 и PS равны и средняя пользовательская оценка жанров Sports и Action равны. Первая гипотеза не подтвердилась, скорее всего оценка пользователей больше зависит от игры, а не от консоли. Вторую гипотезу отвергнуть не получилось, у двух наиболее популярных жанров в среднем и оценки пользователей одинаковые.

# ## 6. Общий вывод: <a id= conclusion></a>
# В данной работе были изучены данные по компьютерным играм с 1980 года до 2016 года. Стремительное развитие рынка наблюдалось с начала 2000х и после 2009 начался спад. В среднем консоль "живет" от 9 до 11 лет, далее она устаревает. Если взять актуальный период с 2012 года( в 2012 году стартовало самое последнее поколение консолей) и до 2015 года( данные за 2016 год могут быть неполными), то рост продаж  наблюдается у 2х консолей: XOne и PS4, они же были выбраны мной как потенциально успешные в 2016 и 2017 году. Среди жанров лидерами являются Action,Sports, Shooter, а также Role Playing, но больше для японских геймеров. Также на продажи влияет рейтинг ESRB: игры для всех, для взрослых и для подростков продаются лучше игр с другим рейтингом. Интересно, что ни оценки критиков, ни оценки пользователей на продажи игр никак не влияют, значит при покупке эти оценки не являются решающими. У жанров Action и Sport одинаковые средние оценки пользователей, что может говорить о потенциале Sports и возможности роста в продажах. 
# 
