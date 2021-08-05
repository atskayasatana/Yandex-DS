#!/usr/bin/env python
# coding: utf-8

# # Самостоятельный проект "ВикиШоп"

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Цель-работы" data-toc-modified-id="Цель-работы-0.1"><span class="toc-item-num">0.1&nbsp;&nbsp;</span>Цель работы</a></span></li><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-0.2"><span class="toc-item-num">0.2&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-0.3"><span class="toc-item-num">0.3&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#Логистическая-регрессия" data-toc-modified-id="Логистическая-регрессия-0.3.1"><span class="toc-item-num">0.3.1&nbsp;&nbsp;</span>Логистическая регрессия</a></span></li></ul></li><li><span><a href="#Метод-опорных-векторов" data-toc-modified-id="Метод-опорных-векторов-0.4"><span class="toc-item-num">0.4&nbsp;&nbsp;</span>Метод опорных векторов</a></span></li><li><span><a href="#Дерево-решений" data-toc-modified-id="Дерево-решений-0.5"><span class="toc-item-num">0.5&nbsp;&nbsp;</span>Дерево решений</a></span></li><li><span><a href="#Результаты-на-тестовой-выборке" data-toc-modified-id="Результаты-на-тестовой-выборке-0.6"><span class="toc-item-num">0.6&nbsp;&nbsp;</span>Результаты на тестовой выборке</a></span></li></ul></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Выводы</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li><li><span><a href="#-Комментарий-ревьюера" data-toc-modified-id="-Комментарий-ревьюера-3"><span class="toc-item-num">3&nbsp;&nbsp;</span> Комментарий ревьюера</a></span></li></ul></div>

# ### Цель работы

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# Необходимо построить модель со значением метрики качества *F1* не меньше 0.75. 
# 
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# ### Подготовка

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers
import pandas as pd
import nltk
import re



from sklearn.metrics import f1_score
from tqdm import notebook
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV 
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC
from sklearn import svm




nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('english'))
warnings.filterwarnings("ignore")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def clear_text(text):
    text1=re.sub(r'[^a-zA-Z ]', ' ', text)
    list_2=text1.split()
    text_2=" ".join(list_2) 
    return text_2


def get_wordnet_pos(word):
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def Lemmatize(text):
    corpus=[]
    lemmatizer = WordNetLemmatizer()
    for i in range(0,len(text)):
        lemma=clear_text(text[i]).lower()
        word_list = nltk.word_tokenize(lemma)
        lemma1=' '.join([lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in word_list])
        corpus.append(lemma1)    
    return corpus


# Загрузим данные из файла в dataframe data и выведем информацию и первые 5 строк.

# In[2]:


data = pd.read_csv('/datasets/toxic_comments.csv')


# In[3]:


data.info()


# In[4]:


data.head()


# В исходном файле 159571 строк и 2 столбца: комментарий и метка(1-токсичный, 0- не токсичный), пропусков нет.

# Преобразуем комментарии в Юникод и лемматизируем их:

# In[5]:


corpus=Lemmatize(list(data['text'].values.astype('U')))


# Создадим выборку features с признаками(текст) и целевым признаком target (токсичность), разобьем их на обучающую, валидационную и тестовую выборки(60/20/20).

# In[6]:


target=data['toxic']

features_train, features_valid_test, target_train, target_valid_test=train_test_split(corpus, target, test_size=0.4)
features_valid, features_test, target_valid, target_test=train_test_split(features_valid_test, target_valid_test, test_size=0.5)


# Преобразуем обучающую выборку с признаками в векторы с помощью TfidfVectorizer'а, теперь признаками будет матрица tf_idf

# In[7]:


count_tf_idf = TfidfVectorizer(stop_words=stopwords) 
tf_idf =count_tf_idf.fit_transform(features_train)


# С помощью transform преобразуем валидационную и тестовые матрицы с признаками:

# In[8]:


tf_idf_valid=count_tf_idf.transform(features_valid)
tf_idf_test=count_tf_idf.transform(features_test)


# ### Обучение

# Рассмотрим 3 модели: логистическую регрессию, метод опорных векторов и дерево

# #### Логистическая регрессия

# In[9]:


lf_model=LogisticRegression(random_state=0, solver='lbfgs')
lf_model.fit(tf_idf, target_train)
lf_prediction=lf_model.predict(tf_idf_valid)
lf_f1=round(f1_score(target_valid, lf_prediction),3)
print('F1 для логистической регрессии до подбора:', lf_f1)


# In[10]:


lf_model.get_params()


# In[11]:


best_lf=LogisticRegression()


# Попробуем с подобрать гиперпараметры, чтобы получить нужный нам f1.

# In[12]:


param_dist = {
              'penalty': ['l1', 'l2'],
              'max_iter': list(range(400,500,100)),
              'solver': ['newton-cg', 'liblinear', 'saga', 'sag','lbfgs'],
               'C':[0.1,0.5,0.99,1]
             }
              
rs = GridSearchCV(best_lf, 
                param_dist, 
                cv = 5, 
                verbose = 1,
                scoring='f1')

rs.fit(tf_idf, target_train)
rs.best_params_


# In[13]:


rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
rs_df.head(5)


# In[14]:


best_lf.set_params(**rs.best_params_)
best_lf.fit(tf_idf, target_train)


# Поверим модель на валидационной выборке

# In[15]:


lf_prediction_valid=best_lf.predict(tf_idf_valid)
lf_f1=round(f1_score(target_valid, lf_prediction_valid),3)
print('F1 для логистической регрессии после подбора на валидационной выборке:', lf_f1)


# ### Метод опорных векторов

# In[16]:


svc_model = svm.SVC(kernel='rbf')


# In[17]:


svc_model.fit(tf_idf, target_train)


# In[18]:


svc_prediction=svc_model.predict(tf_idf_valid)
svc_f1=round(f1_score(target_valid, svc_prediction),3)
print('F1 для МОВ до подбора:', svc_f1)


# In[19]:


svc_model.get_params()


# подберем параметры для модели

# In[20]:


param_dist = {
              'kernel':['rbf'],
              'gamma':['scale'],
              'C':[0.85,1]
    
             }


# In[21]:


best_svc=SVC()


# In[22]:


grid = GridSearchCV(best_svc,param_dist, verbose=1, scoring='f1', cv=3)
grid.fit(tf_idf, target_train)
grid.best_params_


# In[23]:


grid_df = pd.DataFrame(grid.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
grid_df.head()


# In[24]:


best_svc.set_params(**grid.best_params_)
best_svc.fit(tf_idf, target_train)


# In[25]:


cvs_prediction_valid=best_svc.predict(tf_idf_valid)
cvs_f1=round(f1_score(target_valid, cvs_prediction_valid),3)
print('F1 для МОВ валидационной выборке:', cvs_f1)


# ### Дерево решений

# In[26]:


dt_model = DecisionTreeClassifier(random_state=12345)


# In[27]:


dt_model.fit(tf_idf, target_train)


# In[28]:


dt_prediction=dt_model.predict(tf_idf_valid)
dt_f1=round(f1_score(target_valid, dt_prediction),3)
print('F1 для дерева на валидационной выборке:', dt_f1)
dt_model.get_params()


# In[29]:


dt_param = {'max_depth':list(range(50,53,1)),
            'max_leaf_nodes':list(range(5,7,1))                   
    }


# In[30]:


best_dt=DecisionTreeClassifier(random_state=12345)

rt_grid = GridSearchCV(best_dt,dt_param,refit = True,cv=3, verbose=1, scoring='f1')

rt_grid.fit(tf_idf, target_train)

rt_grid.best_params_


# In[31]:


best_dt.set_params(**rt_grid.best_params_)


# In[32]:


best_dt.fit(tf_idf, target_train)
dt_prediction_valid=best_dt.predict(tf_idf_valid)
best_dt_f1=round(f1_score(target_valid, dt_prediction_valid),3)
print('F1 для дерева валидационной выборке:', best_dt_f1)


# ### Результаты на тестовой выборке

# In[33]:


best_lf_prediction_test=best_lf.predict(tf_idf_test)
best_lf_f1=round(f1_score(target_test, best_lf_prediction_test),3)
print('F1 для логистической регрессии на тестовой выборке:', best_lf_f1)


# In[34]:


best_cv_prediction_test=best_svc.predict(tf_idf_test)
best_cv_f1=round(f1_score(target_test, best_cv_prediction_test),3)
print('F1 для МОВ на тестовой выборке:', best_cv_f1)


# In[35]:


best_dt_prediction_test=best_dt.predict(tf_idf_test)
best_dt_f1=round(f1_score(target_test, best_dt_prediction_test),3)
print('F1 для дерева решений на тестовой выборке:', best_dt_f1)


# ## Выводы

# Лучшие результаты при решении задачи тексовой классификации показала логистическая регрессия.Метод опорных векторов также дал высокий F1, но на обучение и подбор гиерпараметров может уйти несколько часов. Дерево классификатор показало худший результат и на валидационеной и на тестовой выборке, для данной модели подобрать гиперпараметры не удалось. Таки образом, можно предположить, что лучшими моделями для решения подобных задач являются линейные классификаторы.

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [ ]  Весь код выполняется без ошибок
# - [ ]  Ячейки с кодом расположены в порядке исполнения
# - [ ]  Данные загружены и подготовлены
# - [ ]  Модели обучены
# - [ ]  Значение метрики *F1* не меньше 0.75
# - [ ]  Выводы написаны
