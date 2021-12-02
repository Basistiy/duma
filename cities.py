#%% Загружаем данные

import numpy as np
import pandas as pd
uiks = pd.read_csv('data/cities_ok.csv', index_col=0)
uiks['total_voters'].sum()

#%% Количество участков в выборке:
len(uiks)
#%% Количество избирателей:
uiks['total_voters'].sum()
#%% Количество избирателей:
uiks['total_voters'].sum()

#%% График зависимости результата на участках от явки

import matplotlib.pyplot as plt
uiks['er_percent'] = uiks['er'] / (uiks['voted'])
uiks['kprf_percent'] = uiks['kprf'] / (uiks['voted'])
er_string = str(round(100*uiks['er'].sum()/uiks['voted'].sum(),2)) + '%'
kprf_string = str(round(100*uiks['kprf'].sum()/uiks['voted'].sum(),2))+ '%'


uiks['turnout'] = uiks['voted']/uiks['total_voters']
plt.scatter(uiks['turnout'], uiks['er_percent'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks['turnout'], uiks['kprf_percent'], color='red', s=0.05, label="КПРФ " + kprf_string)
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#%% Итоговый результат для различных городов

cities = uiks['city300'].drop_duplicates()
city_result = pd.DataFrame()
i=0
for city in cities:
    i+=1
    city_data = uiks[uiks['city300'] == city]
    city_er_percent = round(100*city_data['er'].sum()/city_data['voted'].sum(),2)
    city_kprf_percent = round(100*city_data['kprf'].sum()/city_data['voted'].sum(),2)
    city_result = city_result.append(pd.DataFrame({'name':city,'er_percent': city_er_percent,'kprf_percent': city_kprf_percent}, index=[i]))
city_result.sort_values('er_percent')

#%% Осуществляем вброс

from random import uniform
uiks = uiks.sample(frac=1)

uiks['er_fraud'] = uiks['er']
uiks['kprf_fraud'] = uiks['kprf']
uiks['voted_fraud'] = uiks['voted']
uiks['added'] = False

i = 0
er_percent = uiks['er_fraud'].sum()/uiks['voted_fraud'].sum()
for index, row in uiks.iterrows():
    # 0.47
    if er_percent < 0.47:
        total_voters = row['total_voters']
        voted = row['voted']
        max_fraud = total_voters - voted
        min_fraud = max_fraud*0
        number = int(uniform(min_fraud, max_fraud))
        uiks.loc[index, 'er_fraud'] = row['er'] + number
        uiks.loc[index, 'voted_fraud'] = row['voted'] + number
        uiks.loc[index,'added'] = True
        er_percent = uiks['er_fraud'].sum()/uiks['voted_fraud'].sum()

uiks['turnout_fraud'] = uiks['voted_fraud']/uiks['total_voters']
uiks['er_percent_fraud'] = uiks['er_fraud']/uiks['voted_fraud']
uiks['kprf_percent_fraud'] = uiks['kprf']/uiks['voted_fraud']
er_string = str(round(100*uiks['er_fraud'].sum()/uiks['voted_fraud'].sum(),2)) + '%'
kprf_string = str(round(100*uiks['kprf_fraud'].sum()/uiks['voted_fraud'].sum(),2))+ '%'
plt.scatter(uiks['turnout_fraud'], uiks['er_percent_fraud'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks['turnout_fraud'], uiks['kprf_percent_fraud'], color='red', s=0.05, label="КПРФ " + kprf_string)
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()
uiks['kprf'].sum()/uiks['voted_fraud'].sum()

#%% Осуществляем замену голосов

uiks_change = uiks[~uiks['added']]
uiks['changed'] = False
for index, row in uiks_change.iterrows():
    if er_percent < 0.4982:
        total_voters = row['total_voters']
        random_voted = int(uniform(total_voters * 0.8, total_voters))
        voted = random_voted
        random_er = int(uniform(random_voted * 0.8, random_voted))
        uiks.loc[index, 'voted_fraud'] = voted
        uiks.loc[index, 'er_fraud'] = int(random_er)
        uiks.loc[index, 'kprf_fraud'] = int((random_voted - random_er)*0.3)
        uiks.loc[index, 'changed'] = True
        er_percent = uiks['er_fraud'].sum() / uiks['voted_fraud'].sum()

uiks['turnout_fraud'] = uiks['voted_fraud']/uiks['total_voters']
uiks['er_percent_fraud'] = uiks['er_fraud']/uiks['voted_fraud']
uiks['kprf_percent_fraud'] = uiks['kprf_fraud']/uiks['voted_fraud']

er_string = str(round(100*uiks['er_fraud'].sum()/uiks['voted_fraud'].sum(),2)) + '%'
kprf_string = str(round(100*uiks['kprf_fraud'].sum()/uiks['voted_fraud'].sum(),2))+ '%'
plt.scatter(uiks['turnout_fraud'], uiks['er_percent_fraud'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks['turnout_fraud'], uiks['kprf_percent_fraud'], color='red', s=0.05, label="КПРФ " + kprf_string)
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()
uiks['kprf_fraud'].sum()/uiks['voted_fraud'].sum()

#%% Сохраняем результат фальсификаций в файл

# uiks.to_csv('cities_fraud.csv')

#%% Загружаем фальсификации из файла

uiks = pd.read_csv('data/cities_fraud.csv', index_col=0)

#%% Результат Единой России после фальсификаций

uiks['er_fraud'].sum()/uiks['voted_fraud'].sum()

#%% Результат КПРФ после фальсификаций

uiks['kprf_fraud'].sum()/uiks['voted_fraud'].sum()

#%% Выделяем кластер с нормальной и аномальной явкой

from sklearn.cluster import DBSCAN
er = uiks[['turnout_fraud', 'er_percent_fraud']]
er = er.to_numpy()
db = DBSCAN(eps=0.045, min_samples=200).fit(er)
plt.scatter(er[:, 0], er[:, 1], c=db.labels_, s=0.01)
plt.show()
uiks['db'] = db.labels_
uiks_normal = uiks[uiks['db'] == 0]
uiks_abnormal = uiks[uiks['db'] != 0]
plt.scatter(uiks_normal['turnout_fraud'], uiks_normal['er_percent_fraud'], color='blue', s=0.05)
plt.scatter(uiks_abnormal['turnout_fraud'], uiks_abnormal['er_percent_fraud'], color='red', s=0.05)

plt.xlabel("явка")
plt.ylabel("результат партии")
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

#%% Результат ЕР в ядре

uiks_normal['er_fraud'].sum()/uiks_normal['voted_fraud'].sum()
#%% Результат КПРФ в ядре

uiks_normal['kprf_fraud'].sum()/uiks_normal['voted_fraud'].sum()
#%% Результат ЕР в хвосте

uiks_abnormal['er_fraud'].sum()/uiks_abnormal['voted_fraud'].sum()
#%% Результат КПРФ в хвосте

uiks_abnormal['kprf_fraud'].sum()/uiks_abnormal['voted_fraud'].sum()
#%% Количество участков в ядре

len(uiks_normal)
#%% Количество участков с вбросами в ядре

len(uiks_normal[uiks_normal['added']])
#%% Количество участков с заменой голосов в ядре

len(uiks_normal[uiks_normal['changed']])
#%% Количество участков с вбросами вне ядра

len(uiks_abnormal[uiks_abnormal['added']])

#%% Количество участков с заменой голосов вне ядра

len(uiks_abnormal[uiks_abnormal['changed']])
#%% Создаем pipeline для машинного обучения

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
pipe = Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(weights='distance'))])
pipe.get_params()

#%% Обучаем модель и делаем предсказание

from sklearn.model_selection import GridSearchCV

mod = GridSearchCV(estimator=pipe, param_grid={'model__n_neighbors': [1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,18,19,20]}, cv=3)

X = uiks_normal[['kprf_fraud', 'total_voters', 'lat', 'lon']]
y = uiks_normal['er_fraud']
Xx = uiks_abnormal[['kprf_fraud', 'total_voters', 'lat', 'lon']]
mod.fit(X, y)
prediction = mod.predict(Xx)
uiks_abnormal['er_predicted'] = prediction.round()
pd.DataFrame(mod.cv_results_)

#%% Корректируем результаты, так как знаем, что за Единую Россию не было вбросов

# for index, row in uiks_abnormal.iterrows():
#     if row['er_fraud'] < row['prediction']:
#         uiks_abnormal.loc[index, 'er_predicted'] = row['er_fraud']

#%% Вычисляем явку по результатам машинного обучения

uiks_abnormal['voted_predicted'] = uiks_abnormal['voted_fraud'] - uiks_abnormal['er_fraud'] + uiks_abnormal['er_predicted']
uiks_abnormal['turnout_predicted'] = uiks_abnormal['voted_predicted'] / uiks_abnormal['total_voters']
uiks_abnormal['er_percent_predicted'] = uiks_abnormal['er_predicted'] / uiks_abnormal['voted_predicted']
uiks_abnormal['kprf_percent_predicted'] = uiks_abnormal['kprf'] / uiks_abnormal['voted_predicted']

uiks_normal['er_predicted'] = uiks_normal['er_fraud']
uiks_normal['voted_predicted'] = uiks_normal['voted_fraud']
uiks_normal['turnout_predicted'] = uiks_normal['turnout_fraud']
uiks_normal['er_percent_predicted'] = uiks_normal['er_percent_fraud']
uiks_normal['kprf_percent_predicted'] = uiks_normal['kprf_percent_fraud']

uiks_predicted = uiks_normal.append(uiks_abnormal)
#%% Строим график зависимости результатов на участках со вбросами до фальсификаций
uiks_added = uiks_abnormal[uiks_abnormal['added']]
er_string = str(round(100*uiks_added['er'].sum()/uiks_added['voted'].sum(),2)) + '%'
kprf_string = str(round(100*uiks_added['kprf'].sum()/uiks_added['voted'].sum(),2))+ '%'
plt.scatter(uiks_added['turnout'], uiks_added['er_percent'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks_added['turnout'], uiks_added['kprf_percent'], color='red', s=0.05, label="КПРФ " + kprf_string)
plt.xlim([0, 1])
plt.ylim([0, 1])
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.title("До фальсификаций (участки с вбросами)")
plt.show()

#%% Строим график зависимости результатов на участках от явки со вбросами c фальсификациями
er_string = str(round(100*uiks_added['er_fraud'].sum()/uiks_added['voted_fraud'].sum(),2)) + '%'
kprf_string = str(round(100*uiks_added['kprf_fraud'].sum()/uiks_added['voted_fraud'].sum(),2))+ '%'
plt.scatter(uiks_added['turnout_fraud'], uiks_added['er_percent_fraud'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks_added['turnout_fraud'], uiks_added['kprf_percent_fraud'], color='red', s=0.05, label="КПРФ " + kprf_string)
plt.xlim([0, 1])
plt.ylim([0, 1])
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.title("С фальсификациями (участки с вбросами)")
plt.show()

#%% Строим график зависимости результатов на участках от явки со вбросами после машинного обучения
er_string = str(round(100*uiks_added['er_predicted'].sum()/uiks_added['voted_predicted'].sum(),2)) + '%'
kprf_string = str(round(100*uiks_added['kprf'].sum()/uiks_added['voted_predicted'].sum(),2))+ '%'
plt.scatter(uiks_added['turnout_predicted'], uiks_added['er_percent_predicted'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks_added['turnout_predicted'], uiks_added['kprf_percent_predicted'], color='red', s=0.05, label="КПРФ " + kprf_string)
plt.xlim([0, 1])
plt.ylim([0, 1])
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.title("После машинного обучения (участки с вбросами)")
plt.show()

#%% Строим график зависимости результатов на участках с заменой до фальсификаций
uiks_changed = uiks_abnormal[uiks_abnormal['changed']]
er_string = str(round(100*uiks_changed['er'].sum()/uiks_changed['voted'].sum(),2)) + '%'
kprf_string = str(round(100*uiks_changed['kprf'].sum()/uiks_changed['voted'].sum(),2))+ '%'
plt.scatter(uiks_changed['turnout'], uiks_changed['er_percent'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks_changed['turnout'], uiks_changed['kprf_percent'], color='red', s=0.05, label="КПРФ " + kprf_string)
plt.xlim([0, 1])
plt.ylim([0, 1])
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.title("До фальсификаций (участки с заменой)")
plt.show()

#%% Строим график зависимости результатов на участках от явки с заменой c фальсификациями
er_string = str(round(100*uiks_changed['er_fraud'].sum()/uiks_changed['voted_fraud'].sum(),2)) + '%'
kprf_string = str(round(100*uiks_changed['kprf_fraud'].sum()/uiks_changed['voted_fraud'].sum(),2))+ '%'
plt.scatter(uiks_changed['turnout_fraud'], uiks_changed['er_percent_fraud'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks_changed['turnout_fraud'], uiks_changed['kprf_percent_fraud'], color='red', s=0.05, label="КПРФ " + kprf_string)
plt.xlim([0, 1])
plt.ylim([0, 1])
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.title("С фальсификациями (участки с заменой)")
plt.show()


#%% Строим график зависимости результатов на участках от явки с заменой после машинного обучения
er_string = str(round(100*uiks_changed['er_predicted'].sum()/uiks_changed['voted_predicted'].sum(),2)) + '%'
kprf_string = str(round(100*uiks_changed['kprf_fraud'].sum()/uiks_changed['voted_predicted'].sum(),2))+ '%'
plt.scatter(uiks_changed['turnout_predicted'], uiks_changed['er_percent_predicted'], color='blue', s=0.05, label="Единая Россия " + er_string)
plt.scatter(uiks_changed['turnout_predicted'], uiks_changed['kprf_percent_predicted'], color='red', s=0.05, label="КПРФ " + kprf_string)
plt.xlim([0, 1])
plt.ylim([0, 1])
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.xlabel("явка")
plt.ylabel("результат партии")
plt.show()

#%% Итоговый результат для городов

cities = uiks_predicted['city300'].drop_duplicates()
city_result_predicted = pd.DataFrame()
i=0
for city in cities:
    i+=1
    city_data = uiks_predicted[uiks_predicted['city300'] == city]
    city_er_percent = round(100*city_data['er'].sum()/city_data['voted'].sum(),2)
    city_er_percent_fraud = round(100*city_data['er_fraud'].sum()/city_data['voted_fraud'].sum(),2)
    city_er_percent_predicted = round(100*city_data['er_predicted'].sum()/city_data['voted_predicted'].sum(),2)
    er_error = city_er_percent_predicted - city_er_percent
    city_result_predicted = city_result_predicted.append(pd.DataFrame({'name':city,'er_percent_fraud': city_er_percent_fraud,'er_percent_predicted': city_er_percent_predicted, 'er_percent': city_er_percent,'er_error': er_error}, index=[i]))
city_result_predicted.sort_values('er_percent')

#%% Максимальная ошибка Единая Россия

city_result_predicted['er_error'].max()

#%% Максимальная ошибка Единая Россия

city_result_predicted['er_error'].min()
#%% Строим график зависимости результатов городов сгенерированных от первоначальных

plt.scatter(city_result_predicted['er_percent'], city_result_predicted['er_percent'], color='red', label = "Истинные результаты")
plt.scatter(city_result_predicted['er_percent'], city_result_predicted['er_percent_predicted'], color='blue', label = "Результаты машинного обучения")
lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
plt.xlabel("Результат ЕР в городах")
plt.ylabel("Результат ЕР в городах")

plt.show()

# fraud_cor = np.corrcoef(city_result_predicted['er_percent'], city_result_predicted['er_percent_fraud'])
# plt.scatter(city_result_predicted['er_percent'], city_result_predicted['er_percent_fraud'], color='red')
#
# plt.show()
#

