import warnings
import pandas as pd
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.read_csv('../dataset.csv')
df_new_idf1 = pd.read_csv('../dataset_2_idf1_2min.csv')

df_new_CV1 = df_new_idf1[df_new_idf1.columns[df_new_idf1.columns.str.startswith('CV1')].tolist()]


df_new_CV1['date_time'] = pd.to_datetime(df_new_idf1['Unnamed: 0'].apply(lambda x: x.split('+')[0]), format='%Y-%m-%d %H:%M:%S')
df_new_CV1 = df_new_CV1.set_index('date_time')
df_new_CV1 = df_new_CV1.resample('2min').bfill()

df_new_idf2 = pd.read_csv('../dataset_2_idf2_2min_complete.csv')

df_new_CV2 = df_new_idf2[df_new_idf2.columns[df_new_idf2.columns.str.startswith('CV2')].tolist()]


df_new_CV2['date_time'] = pd.to_datetime(df_new_idf2['timestamp'].apply(lambda x: x.split('+')[0]), format='%Y-%m-%d %H:%M:%S')
df_new_CV2 = df_new_CV2.set_index('date_time')

df['date_time'] = pd.to_datetime(df['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S')
df = df.drop(columns=['Unnamed: 0'])

df = df.set_index('date_time')


translation_dict = {
    'XXXXXXXXXXX'
}

reversed_dict = dict(map(reversed, translation_dict.items()))
# Split by IDF and correct outliers


dfCV1 = df[df.columns[df.columns.str.startswith('CV1')].tolist()]
dfCV2 = df[df.columns[df.columns.str.startswith('CV2')].tolist()]

dfCV1 = pd.concat([dfCV1, df_new_CV1], axis=0)
dfCV1 = dfCV1.drop(columns=['XXXXXXXXXXX'])

dfCV2 = pd.concat([dfCV2, df_new_CV2], axis=0)
dfCV2 = dfCV2.drop(columns=['XXXXXXXXXXX'])


vibrations = [['XXXXXXXXXXX'],
              ['XXXXXXXXXXX']]
# Gas sampling except H2
gas_sampling = [['XXXXXXXXXXX'],
                ['XXXXXXXXXXX']]
# Tº for stators + gas
temperatures = [['XXXXXXXXXXX'] + ['XXXXXXXXXXX'],
                ['XXXXXXXXXXX'] + ['XXXXXXXXXXX']]
oil_fan = [['XXXXXXXXXXX'],
           ['XXXXXXXXXXX']]
current = [['XXXXXXXXXXX']]
oil_level = [['XXXXXXXXXXX']]
oil_motor = [['XXXXXXXXXXX'],
             ['XXXXXXXXXXX']]
speeds = [['XXXXXXXXXXX'],
          ['XXXXXXXXXXX']]
displ_percent = [['XXXXXXXXXXX'], ['XXXXXXXXXXX']]

ge_zero1 = vibrations[0] + temperatures[0] + oil_fan[0] + current[0] + oil_level[0] + \
    oil_motor[0] + ['XXXXXXXXXXX'] + \
    speeds[0] + displ_percent[0]
ge_zero2 = vibrations[1] + temperatures[1] + oil_fan[1] + current[1] + oil_level[1] + \
    oil_motor[1] + ['XXXXXXXXXXX'] + \
    speeds[1] + displ_percent[1]
for col1, col2 in zip(ge_zero1, ge_zero2):
    dfCV1.loc[dfCV1[col1] < 0, col1] = 0
    dfCV2.loc[dfCV2[col2] < 0, col2] = 0
for col1, col2 in zip(oil_fan[0], oil_fan[1]):
    dfCV1.loc[dfCV1[col1] > 40, col1] = 40
    dfCV2.loc[dfCV2[col2] > 40, col2] = 40
for col1, col2 in zip(temperatures[0], temperatures[1]):
    dfCV1.loc[dfCV1[col1] >= 155, col1] = 155
    dfCV2.loc[dfCV2[col2] >= 155, col2] = 155
for col1, col2 in zip(oil_motor[0], oil_motor[1]):
    dfCV1.loc[dfCV1[col1] > 10, col1] = 10
    dfCV2.loc[dfCV2[col2] > 10, col2] = 10
dfCV1.loc[dfCV1['XXXXXXXXXXX'] < -1, 'XXXXXXXXXXX'] = 0
dfCV2.loc[dfCV2['XXXXXXXXXXX'] < -1, 'XXXXXXXXXXX'] = 0
for col1, col2 in zip(displ_percent[0], displ_percent[1]):
    dfCV1.loc[dfCV1[col1] > 100, col1] = 100
    dfCV2.loc[dfCV2[col2] > 100, col2] = 100
