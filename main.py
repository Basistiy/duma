#%%

from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


uiks = pd.read_csv('data/eldata.csv')
kprf = uiks['kprf'].sum()/uiks['voted'].sum()
er = uiks['er'].sum()/uiks['voted'].sum()
uiks['er_percent'] = uiks['er'] / (uiks['voted'])
uiks['kprf_percent'] = uiks['kprf'] / (uiks['voted'])
#%%
import matplotlib.pyplot as plt
uiks = uiks[uiks['kprf']>10]
uiks = uiks[uiks['er']>10]
uiks['er_percent'] = uiks['er'] / (uiks['voted'])
uiks['kprf_percent'] = uiks['kprf'] / (uiks['voted'])
uiks['turnout'] = uiks['voted']/uiks['total_voters']
plt.scatter(uiks['turnout'], uiks['er_percent'], color='blue', s=0.01)
plt.scatter(uiks['turnout'], uiks['kprf_percent'], color='red', s=0.01)
plt.show()
#%%
from sklearn.cluster import DBSCAN
uiks = uiks[uiks['kprf']>10]
uiks = uiks[uiks['er']>10]
er = uiks[['turnout', 'er_percent']]
er = er.to_numpy()
db = DBSCAN(eps=0.009, min_samples=175).fit(er)

plt.scatter(er[:, 0], er[:, 1], c=db.labels_, s=0.01)
plt.show()
#%%

uiks['db'] = db.labels_
uiks_normal = uiks[uiks['db'] == 0]
uiks_abnormal = uiks[uiks['db'] != 0]
plt.scatter(uiks_normal['turnout'], uiks_normal['er_percent'], color='blue', s=0.01)
plt.show()
plt.scatter(uiks_abnormal['turnout'], uiks_abnormal['er_percent'], color='blue', s=0.01)
plt.show()
uiks_normal['total_voters'].sum()
#%%
uiks['color'] = ''
regions = uiks['region'].drop_duplicates()
reg = pd.DataFrame()
for region in regions:
   region_data = uiks[uiks['region'] == region]
   kprf_total = region_data['kprf'].sum()
   er_total = region_data['er'].sum()
   if er_total>kprf_total:
       uiks.loc[uiks['region'] == region, 'color'] = 'blue'
   else:
       uiks.loc[uiks['region'] == region, 'color'] = 'red'
   reg = reg.append(pd.DataFrame({'name': region,'kprf':[kprf_total],'er':[er_total]}), ignore_index=True)


kprf_wins = [reg[reg['kprf']>reg['er']]]
print(kprf_wins)
#%%

kprf_percent_normal = uiks_normal['kprf'].sum() / (
        uiks_normal['voted'].sum())
er_percent_normal = uiks_normal['er'].sum() / (uiks_normal['voted'].sum())
len(uiks_normal)
uiks_normal_total_voters = uiks_normal['total_voters'].sum()
plt.scatter(uiks_normal['turnout'], uiks_normal['er_percent'], color='blue', s=0.01)
plt.scatter(uiks_normal['turnout'], uiks_normal['kprf_percent'], color='red', s=0.01)
plt.show()
#%%

uiks_normal['kprf_total_percent'] = uiks_normal['kprf'] / uiks_normal['total_voters']
uiks_normal['er_total_percent'] = uiks_normal['er'] / uiks_normal['total_voters']
uiks_abnormal['kprf_total_percent'] = uiks_abnormal['kprf'] / uiks_abnormal['total_voters']
pipe = Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())])
pipe.get_params()
#%%

mod = GridSearchCV(estimator=pipe, param_grid={'model__n_neighbors': [30, 35, 40, 45, 50, 55]}, cv=3)
X = uiks_normal[['kprf', 'total_voters', 'lat', 'lon']]
y = uiks_normal['er']
Xx = uiks_abnormal[['kprf', 'total_voters', 'lat', 'lon']]
mod.fit(X, y)
prediction = mod.predict(Xx)
uiks_abnormal['prediction'] = prediction
uiks_abnormal['er_predicted'] = prediction.round()
pd.DataFrame(mod.cv_results_)
#%%

for index, row in uiks_abnormal.iterrows():
    if row['er'] < row['prediction']:
        uiks_abnormal.loc[index, 'er_predicted'] = row['er']
uiks_normal['er_predicted'] = uiks_normal['er']

#%%

uiks_abnormal['voted_predicted'] = uiks_abnormal['voted'] - uiks_abnormal['er'] + uiks_abnormal['er_predicted']
uiks_normal['voted_predicted'] = uiks_normal['voted']
uiks_abnormal['turnout_predicted'] = uiks_abnormal['voted_predicted'] / uiks_abnormal['total_voters']
uiks_normal['turnout_predicted'] = uiks_normal['turnout']
uiks_abnormal['er_percent_predicted'] = uiks_abnormal['er_predicted'] / uiks_abnormal['voted_predicted']
uiks_normal['er_percent_predicted'] = uiks_normal['er_percent']
uiks_abnormal['kprf_percent_predicted'] = uiks_abnormal['kprf'] / uiks_abnormal['voted_predicted']
uiks_normal['kprf_percent_predicted'] = uiks_normal['kprf_percent']

uiks_predicted = uiks_normal.append(uiks_abnormal)
plt.scatter(uiks_predicted['turnout_predicted'], uiks_predicted['er_percent_predicted'], color='blue', s=0.01)
plt.scatter(uiks_predicted['turnout_predicted'], uiks_predicted['kprf_percent_predicted'], color='red', s=0.01)
plt.show()
er_real = uiks_normal['er'].sum() + uiks_abnormal['er_predicted'].sum()
kprf_real = uiks_normal['kprf'].sum() + uiks_abnormal['kprf'].sum()
total_voters_real = uiks_normal['voted'].sum() + uiks_abnormal[
    'voted'].sum()
total_voters_real
er_real_percent = er_real / total_voters_real
er_real_percent
kprf_real_percent = kprf_real / total_voters_real
kprf_real_percent

#%%
reg_true = pd.DataFrame()

for region in regions:
   region_data = uiks_predicted[uiks_predicted['region'] == region]
   kprf_total = region_data['kprf'].sum()
   er_total = region_data['er_predicted'].sum()
   if er_total>kprf_total:
       uiks_predicted.loc[uiks['region'] == region, 'color'] = 'blue'
   else:
       uiks_predicted.loc[uiks['region'] == region, 'color'] = 'red'
   reg_true = reg_true.append(pd.DataFrame({'name': region,'kprf':[kprf_total],'er':[er_total]}), ignore_index=True)



