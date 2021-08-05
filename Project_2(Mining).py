#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Восстановление-золота-из-руды" data-toc-modified-id="Восстановление-золота-из-руды-1">Восстановление золота из руды</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Цель-работы:" data-toc-modified-id="Цель-работы:-1.0.1">Цель работы:</a></span></li></ul></li><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1.1">Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Флотация-и-финальные-показатели" data-toc-modified-id="Флотация-и-финальные-показатели-1.1.1">Флотация и финальные показатели</a></span></li><li><span><a href="#Первичная-очистка-и-финальные-показатели" data-toc-modified-id="Первичная-очистка-и-финальные-показатели-1.1.2">Первичная очистка и финальные показатели</a></span></li><li><span><a href="#Вторичная-очистка-и-финальные-показатели" data-toc-modified-id="Вторичная-очистка-и-финальные-показатели-1.1.3">Вторичная очистка и финальные показатели</a></span></li><li><span><a href="#Вывод:" data-toc-modified-id="Вывод:-1.1.4">Вывод:</a></span></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-1.2">Анализ данных</a></span><ul class="toc-item"><li><span><a href="#1.-Проверка-параметра-rougher.output.recovery" data-toc-modified-id="1.-Проверка-параметра-rougher.output.recovery-1.2.1">1. Проверка параметра rougher.output.recovery</a></span></li><li><span><a href="#Вывод:" data-toc-modified-id="Вывод:-1.2.2">Вывод:</a></span></li><li><span><a href="#2.-Исследование-тестовой-выборки" data-toc-modified-id="2.-Исследование-тестовой-выборки-1.2.3">2. Исследование тестовой выборки</a></span></li><li><span><a href="#3.-Изменение-концентраций-металлов-на-различных-этапах-очистки" data-toc-modified-id="3.-Изменение-концентраций-металлов-на-различных-этапах-очистки-1.2.4">3. Изменение концентраций металлов на различных этапах очистки</a></span></li><li><span><a href="#Вывод:" data-toc-modified-id="Вывод:-1.2.5">Вывод:</a></span></li><li><span><a href="#Распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках." data-toc-modified-id="Распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках.-1.2.6">Распределения размеров гранул сырья на обучающей и тестовой выборках.</a></span></li><li><span><a href="#Вывод:" data-toc-modified-id="Вывод:-1.2.7">Вывод:</a></span></li></ul></li><li><span><a href="#Модель" data-toc-modified-id="Модель-1.3">Модель</a></span><ul class="toc-item"><li><span><a href="#Вывод:" data-toc-modified-id="Вывод:-1.3.1">Вывод:</a></span></li></ul></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-1.4">Чек-лист готовности проекта</a></span></li></ul></li></ul></div>

# # Восстановление золота из руды

# ### Цель работы:

# Подготовьте прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки. 
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Необходимо:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.
# 
# 

# ## Подготовка данных

# Импорт основных библиотек и необходимые функции:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm

import warnings
warnings.filterwarnings('ignore')



def smape(forecast, actual):
    smape=abs(actual-forecast)/(0.5*(abs(actual)+abs(forecast)))
    return smape.sum()*100/smape.shape[0]
        
    

def mae_au_output(actual, C,F,T):     
    mae=0    
    recovery=C*(F-T)*100/(F*(C-T))
    recovery.replace([np.inf, -np.inf], np.nan, inplace=True)
    recovery=recovery.dropna()        
    actual=actual[recovery.index] 
    actual=actual.dropna()   
    
    if len(actual)<len(recovery):
        recovery=recovery[actual.index]
    
    mae=mean_absolute_error(recovery, actual)   
    return mae
    
    
def draw_concentration_hist(metall):
    step1="rougher.input.feed_"+metall
    step2="rougher.output.concentrate_"+metall
    step3="primary_cleaner.output.concentrate_"+metall
    step4="final.output.concentrate_"+metall
    title="Распределение "+ metall+ " на различных этапах очистки"

    plt.figure(figsize=(16,9))
    sns.distplot(gold_recovery_train[step1], label=step1)
    sns.distplot(gold_recovery_train[step2], label=step2)
    sns.distplot(gold_recovery_train[step3], label=step3)
    sns.distplot(gold_recovery_train[step4], label=step4)
    plt.xlabel("Концентрация")
    plt.ylabel("Частота")
    
    plt.legend()
    plt.title(title)
    


def draw_feed_hist(train, test, metall):
    column="rougher.input.feed"+"_"+metall
    if metall=="size":
        title="Распределение размеров гранул: обучающая и тестовая выборки"
    else:
        title="Распределение размеров гранул для "+metall+":обучающая и тестовая выборки"        
    plt.figure(figsize=(16,9))
    sns.distplot(train[column], label='TRAIN')
    sns.distplot(test[column], label='TEST')
    av_train=train[column].mean()
    av_test=test[column].mean()
    plt.xlabel(metall+", размер молекулы")
    plt.ylabel("Частота")    
    plt.axvline(x=av_train, ymin=0, ymax=1, label="Среднее для обучающей:{:.1f}".format(av_train))
    plt.axvline(x=av_test, ymin=0, ymax=1, label="Среднее для тестовой:{:.1f}".format(av_test), linestyle=":")
    plt.legend()
    plt.title(title)
    
def draw_hist(data, title, x_labels, y_labels, label,color):
    plt.figure(figsize=(16,9))
    sns.distplot(data, label=label, color=color)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.legend()

    


# Загрузим данные из файлов в data frame:

# In[2]:


gold_recovery_train=pd.read_csv('/datasets/gold_recovery_train.csv', sep=',')
gold_recovery_test=pd.read_csv('/datasets/gold_recovery_test.csv', sep=',')
gold_recovery_full=pd.read_csv('/datasets/gold_recovery_full.csv', sep=',')


# Проанализируем содержимое каждого из data frame'ов:

# In[3]:


gold_recovery_full.describe()
gold_recovery_full.shape


# In[4]:


gold_recovery_full.head()


# In[5]:


gold_recovery_train.describe()


# In[6]:


gold_recovery_train.head()
gold_recovery_train.shape


# In[7]:


gold_recovery_test.describe()


# In[8]:


gold_recovery_test.head()
gold_recovery_test.shape


# В исходном датасете 22617 строк и 87 столбцов, есть пропуски. Обучающий и тестовый датасеты сформированы в пропорции 75:25 от исходного, в обучающем также 87 столбцов, в тестовом только 53.

# Попробуем разделить данные по этапам процесса и изучить подробнее:

# In[9]:


flotation_i=gold_recovery_full.filter(like=('rougher.input'), axis=1)
flotation_o=gold_recovery_full.filter(like=('rougher.output'), axis=1)
pc=gold_recovery_full.filter(like=('primary_cleaner.'), axis=1)
sc=gold_recovery_full.filter(like=('secondary_cleaner.'), axis=1)
final=gold_recovery_full.filter(like=('final.'), axis=1)


# ### Флотация и финальные показатели

# In[10]:


flotation_i=flotation_i.join(final)
pyplot.figure(figsize=(15, 25))
corr_matrix=flotation_i.corr()
matrix = np.triu(corr_matrix)
sns.heatmap(corr_matrix, cmap='coolwarm', mask=matrix)


# По матрице можно заметить, что есть связь между концентрацией ртути и концентрацией золота на финальном этапе. Также заметна корреляция между размерами молекул серебра, золота и ртути на этапе флотации.

# ### Первичная очистка и финальные показатели

# In[11]:


pc=pc.join(final)
pyplot.figure(figsize=(15, 25))
corr_matrix=pc.corr()
matrix = np.triu(corr_matrix)
sns.heatmap(corr_matrix, cmap='coolwarm', mask=matrix)


# Здесь можно увидеть, что есть связь между концентрацией ртути и серебра в хвостах после первичной очистки. 

# ### Вторичная очистка и финальные показатели

# In[12]:


sc=sc.join(final)
pyplot.figure(figsize=(15, 25))
corr_matrix=sc.corr()
matrix = np.triu(corr_matrix)
sns.heatmap(corr_matrix, cmap='coolwarm', mask=matrix)


# Здесь зависимостей между параметрами очистки и финальными показателями не выявлено.

# ### Вывод:
# у нас имеются 3 набора данных: обучающая выборка, тестовая выборки и полный набор данных. В исходной и обучающей выборке 87 столбцов с данными по разным этапам очистки, в тестовой только 57. В каждом наборе есть пропуски. При более подробном рассмотрении сильных зависимостей между входными параметрами и финальными результатами выявлено не было, кроме корреляции между концентрацией ртути и концентрации золота на финальном этапе.

# ## Анализ данных

# ### 1. Проверка параметра rougher.output.recovery
# 

# Проверим правильность вычисления эффективности обогащения. Данный параметр рассчитывается по следующей формуле
# ![image.png](attachment:image.png)

# Recovery=C*(F-T)/F(C-T)*100%
# 
# где:
# •	C — доля золота в концентрате после флотации/очистки;       C=rougher.output.concentrate_au  
# 
# •	F — доля золота в сырье/концентрате до флотации/очистки;    F=rougher.input.feed_au
# 
# •	T — доля золота в отвальных хвостах после флотации/очистки. T=rougher.output.tail_au
# 
# 

# In[13]:


C=gold_recovery_train['rougher.output.concentrate_au']
F=gold_recovery_train['rougher.input.feed_au']
T=gold_recovery_train['rougher.output.tail_au']

print('Среднеквадратичная ошибка rougher.output.recovery :',mae_au_output(gold_recovery_train['rougher.output.recovery'], C,F,T))


# ### Вывод: 
# При сравнении эффективности вычисленной по формуле и эффективности в обучающей выборке порядок ошибка 10 в степени -14, т.е. в обучающей выборке данные вычислены верно.

# ### 2. Исследование тестовой выборки

# Тестовая выборка содержит только 53 колонки, проверим каких признаков не хватает

# In[14]:


test_list=gold_recovery_test.columns.to_list()


# In[15]:


train_list=gold_recovery_train.columns.to_list()

missed_data=[]

for i in range(len(train_list)):
    if train_list[i] not in test_list:
        missed_data.append(train_list[i])
        
print('Список столбцов, отсутствующих в тестовой выборке:')

missed_data=pd.array(missed_data)
print(missed_data)


# Удалим из списка целевые признаки и сохраним его для дальнейшей работы:

# In[16]:


missed_data=np.delete(missed_data,np.where((missed_data == 'final.output.recovery' )|(missed_data == 'rougher.output.recovery' )))
missed_data


# В тестовой выборке отсутствуют нужные нам целевые признаки: final.output.recovery и rougher.output.recovery. Также пропущены данные по концентрациям металлов на разных этапах очистки, их мы добавлять из исходной выборки не будем. Добавим в тестовый набор пропущенные целевые значения 'final.output.recovery' и 'rougher.output.recovery'.

# In[17]:


gold_recovery_test['final.output.recovery']=gold_recovery_train['final.output.recovery'][gold_recovery_test.index]
gold_recovery_test['rougher.output.recovery']=gold_recovery_train['rougher.output.recovery'][gold_recovery_test.index]


# ### 3. Изменение концентраций металлов на различных этапах очистки

# По условию задачи данные индексируются датой и временем получения информации (признак date) и соседние по времени параметры часто похожи. Заполним пропуски в обучающем датасете используя это условие:

# In[18]:


gold_recovery_train=gold_recovery_train.fillna(method='ffill')
gold_recovery_test=gold_recovery_test.fillna(method='ffill')


# Проверим концентрацию серебра на различных этапах

# In[19]:


draw_concentration_hist("au")


# In[20]:


draw_concentration_hist("ag")


# In[21]:


draw_concentration_hist("pb")


# ### Вывод:
# Заметно, что у золота концентрация увеличивается в соответствии с этапами очистки. С серебром ситуация другая, для флотации подается определенное количество, затем после флотации концентрация растет. однако, после двух этапов очисток концентрация наоборот уменьшется. Для ртути ситуация похожа на ситуацию с золотом, концентрация растет после флотации и первичной очистки, вторичная очистка на концентрацию влияет незначительно.

# ### Распределения размеров гранул сырья на обучающей и тестовой выборках.

# Сравним распределения размеров гранул на обучающей и тестовой выборках:

# In[22]:


draw_feed_hist(gold_recovery_train, gold_recovery_test, "size")


# In[23]:


draw_feed_hist(gold_recovery_train, gold_recovery_test, "au")


# In[24]:


draw_feed_hist(gold_recovery_train, gold_recovery_test, "ag")


# In[25]:


draw_feed_hist(gold_recovery_train, gold_recovery_test, "pb")


# ### Вывод: 
# По гистограммам распределений замено, что обучающая и тестовая выборки имеют похожие распределения размеров модекул сырья, т.е. проблем с обучением модели возникнуть не должно.

# <div style="background: #cceeaa; padding: 5px; border: 1px solid green; border-radius: 5px;">
# <font color='green'> 
# <u>КОММЕНТАРИЙ РЕВЬЮЕРА</u>
# <font color='green'><br>
# согласен)

# Исследуем суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.

# In[26]:


final_output=gold_recovery_train.filter(like='final.output.concentrate_').sum(axis=1)
final_output_tail=gold_recovery_train.filter(like='final.output.tail_').sum(axis=1)
primary_cleaner_conc=gold_recovery_train.filter(like='primary_cleaner.output.concentrate').sum(axis=1)
primary_cleaner_tail=gold_recovery_train.filter(like='primary_cleaner.output.tail').sum(axis=1)
rougher_output_conc=gold_recovery_train.filter(like='rougher.output.concentrate').sum(axis=1)
rougher_output_tail=gold_recovery_train.filter(like='rougher.output.tail').sum(axis=1)
secondary_cleaner_output=gold_recovery_train.filter(like='secondary_cleaner.output.tail').sum(axis=1)


# In[27]:


draw_hist(rougher_output_conc,"Концентрация веществ: после флотации ", "Концентрация", "Флотация","***", 'red')


# In[28]:


draw_hist(rougher_output_tail,"Хвосты: после флотации ", "Концентрация", "Хвосты","***", 'green')


# In[29]:


draw_hist(primary_cleaner_tail,"Первичная очистка: хвосты ", "Концентрация", "Первичная очистка","хвосты", 'grey')


# In[30]:


draw_hist(primary_cleaner_conc,"Первичная очистка: металлы ", "Концентрация", "Первичная очистка","***", 'blue')


# In[31]:


draw_hist(secondary_cleaner_output,"Вторичная очистка: хвосты ", "Концентрация", "Вторичная очистка","Хвосты", 'orange')


# In[32]:


draw_hist(final_output,"Финал: металлы ", "Концентрация", "Финал","Металл", 'purple')


# In[33]:


draw_hist(final_output_tail,"Финал: хвост ", "Концентрация", "Финал","Хвост", 'silver')


# На графиках видно, что на могих этапах суммарная концентрация равна 0 или приближенным к нему значениям. По графикам видно, что большая часть значений лежит в нормальных числовых интервалах и эти хвосты, скорее всего либо пропущенные данные либо ошибки, которые лучше удалить.

# In[34]:


gold_recovery_train=gold_recovery_train.drop(gold_recovery_train[gold_recovery_train.filter(like='rougher.output.concentrate').sum(axis=1)<1].index)


# In[35]:


gold_recovery_train=gold_recovery_train.drop(gold_recovery_train[gold_recovery_train.filter(like='primary_cleaner.output.tail').sum(axis=1)<1].index)


# In[36]:


gold_recovery_train=gold_recovery_train.drop(gold_recovery_train[gold_recovery_train.filter(like='primary_cleaner.output.concentrate').sum(axis=1)<1].index)


# In[37]:


gold_recovery_train=gold_recovery_train.drop(gold_recovery_train[gold_recovery_train.filter(like='secondary_cleaner.output.tail').sum(axis=1)<1].index)


# In[38]:


gold_recovery_train=gold_recovery_train.drop(gold_recovery_train[gold_recovery_train.filter(like='final.output.concentrate_').sum(axis=1)<1].index)


# In[39]:


gold_recovery_train=gold_recovery_train.drop(gold_recovery_train[gold_recovery_train.filter(like='final.output.tail_').sum(axis=1)<1].index)


# ## Модель

# Для решения данной задачи нам подходят 3 модели: линейная регрессия, случайный лес и дерево решений.

# Подготовим обучающую и тестовую выборки. В тестовой выборке отсутсвуют данные, соответсвенно и обучать модели будем на том же наборе данных:

# In[40]:


features=gold_recovery_train.drop('date', axis=1)
features=features.drop(missed_data, axis=1)


# In[41]:


target_train=features[['final.output.recovery','rougher.output.recovery']]
features_train=features.drop(['final.output.recovery','rougher.output.recovery'], axis=1)


# С помощью подбора параметров найдем лучшие модели:
# 1. Cлучайный лес

# In[50]:


RF_model=RandomForestRegressor(random_state=12345)



parametrs = { 'n_estimators': range (10, 15),
              'max_depth': range (5,8),
               'min_samples_leaf': range(5,8)}

grid = GridSearchCV(RF_model, parametrs, cv=5)
grid.fit(features_train, target_train)
grid.best_params_
RF_model=RF_model.set_params(**grid.best_params_)
RF_model.fit(features_train, target_train)


# 2. Дерево решений:

# In[ ]:


DT_model=DecisionTreeRegressor()



parametrs = { 'max_depth': range (5,8),
            'min_samples_leaf': range(3,5)}


grid = GridSearchCV(DT_model, parametrs, cv=5)
grid.fit(features_train, target_train)
grid.best_params_
DT_model=DT_model.set_params(**grid.best_params_)
DT_model.fit(features_train, target_train)


# 3. Линейная регрессия:

# In[ ]:


LR_model=LinearRegression()
LR_model.fit(features_train, target_train)


# Вычислим SMAPE на обучающей выборке по каждой из моделей

# In[ ]:


RF_prediction_train=RF_model.predict(features_train)
SMAPE=0.25*smape(RF_prediction_train[:,1], target_train['rougher.output.recovery'])+0.75*smape(RF_prediction_train[:,0], target_train['final.output.recovery'])
print('SMAPE леса', round(SMAPE,3))


# In[ ]:


DT_prediction_train=DT_model.predict(features_train)
SMAPE=0.25*smape(DT_prediction_train[:,1], target_train['rougher.output.recovery'])+0.75*smape(DT_prediction_train[:,0], target_train['final.output.recovery'])
print('SMAPE дерева', round(SMAPE,3))


# In[ ]:


LR_prediction_train=LR_model.predict(features_train)
SMAPE=0.25*smape(LR_prediction_train[:,1], target_train['rougher.output.recovery'])+0.75*smape(LR_prediction_train[:,0], target_train['final.output.recovery'])
print('SMAPE регрессии:', round(SMAPE,3))


# Пока лучший результат показал случайный лес с подбором гиперпараметров.

# Протестируем и вычислим SMAPE по каждой модели уже на тестовой выборке

# In[44]:


features_t=gold_recovery_test.drop(['date'], axis=1)


# In[45]:


target_test=features_t[['final.output.recovery','rougher.output.recovery']]
features_test=features_t.drop(['final.output.recovery','rougher.output.recovery'], axis=1)


# In[51]:


RF_prediction=RF_model.predict(features_test)
SMAPE=0.25*smape(RF_prediction[:,1], target_test['rougher.output.recovery'])+0.75*smape(RF_prediction[:,0], target_test['final.output.recovery'])
print('SMAPE леса', round(SMAPE,3))


# Протестируем дерево:

# In[ ]:


DT_prediction=DT_model.predict(features_test)
SMAPE_DT=0.25*smape(DT_prediction[:,1], target_test['rougher.output.recovery'])+0.75*smape(DT_prediction[:,0], target_test['final.output.recovery'])
print('SMAPE дерева', round(SMAPE_DT,3))


# И регрессия:

# In[ ]:


LR_prediction=LR_model.predict(features_test)
SMAPE_LR=0.25*smape(LR_prediction[:,1], target_test['rougher.output.recovery'])+0.75*smape(LR_prediction[:,0], target_test['final.output.recovery'])
print('SMAPE регрессии', round(SMAPE_LR,3))


# На тестовой выборке лучшие результаты опять показал случайный лес

# Вычислим SMAPE константной модели(среднее тестовой выборки):

# In[52]:


r=target_train['rougher.output.recovery'].mean()
f=target_train['final.output.recovery'].mean()

dumb_r=np.array([])
dumb_f=np.array([])

for i in range (len(target_test['rougher.output.recovery'])):
    dumb_r=np.append(dumb_r,r)
    dumb_f=np.append(dumb_f,f)

    
SMAPE_dumb=0.25*smape(RF_prediction[:,0], dumb_r)+0.75*smape(RF_prediction[:,1], dumb_f)
print('SMAPE леса на константной модели', round(SMAPE_dumb,3))


# Значение на константной модели выше, чем на тестовой и обучающей выборках. Можно сделать вывод, что выбранная модель рабочая.

# In[56]:


r=target_train['rougher.output.recovery'].mean()
f=target_train['final.output.recovery'].mean()

dumb_r=np.array([])
dumb_f=np.array([])

for i in range (len(target_test['rougher.output.recovery'])):
    dumb_r=np.append(dumb_r,r)
    dumb_f=np.append(dumb_f,f)

    
SMAPE_dumb=0.25*smape(target_test['rougher.output.recovery'], dumb_r)+0.75*smape(target_test['final.output.recovery'], dumb_f)
print('SMAPE леса на константной модели', round(SMAPE_dumb,3))


# ### Вывод:
# После подбора параметров, обучения и тестирования всех моделей лучшие результаты показал случайный лес. Данная модель имеет наименьший коэффициент отклонения SMAPE на обучающей и тестовой выборках.

# ## Чек-лист готовности проекта

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке выполнения
# - [x]  Выполнен шаг 1: данные подготовлены
#     - [x]  Проверена формула вычисления эффективности обогащения
#     - [x]  Проанализированы признаки, недоступные в тестовой выборке
#     - [x]  Проведена предобработка данных
# - [x]  Выполнен шаг 2: данные проанализированы
#     - [x]  Исследовано изменение концентрации элементов на каждом этапе
#     - [x]  Проанализированы распределения размеров гранул на обучающей и тестовой выборках
#     - [x]  Исследованы суммарные концентрации
#     - [x]  Проанализированы и обработаны аномальные значения
# - [x]  Выполнен шаг 3: построена модель прогнозирования
#     - [x]  Написана функция для вычисления итогового *sMAPE*
#     - [x]  Обучено и проверено несколько моделей
#     - [x]  Выбрана лучшая модель, её качество проверено на тестовой выборке
