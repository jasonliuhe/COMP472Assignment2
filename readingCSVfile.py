import pandas as pd


csv_data = pd.read_csv('Data/hns_2018_2019.csv')

data_2018 = csv_data.loc[csv_data['year'] == 2018]
data_2019 = csv_data.loc[csv_data['year'] == 2019]
print(data_2018.shape)
print(data_2018)
print(data_2019.shape)
print(data_2019)
