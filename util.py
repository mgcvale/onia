import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data(test_size=0.2):
    df = pd.read_csv('treino.csv')
    df.drop('id', axis=1, inplace=True)
    x = df.drop('target', axis=1)
    x.drop('col_10', inplace=True, axis=1)
    y = df['target']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=1234)
    return x_train, x_test, y_train, y_test

"""
df = pd.read_csv('treino.csv')
y = df['target']
plt.figure(figsize=(8, 6))
plt.hist(y, bins=np.arange(6)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(5), ['Planeta Deserto', 'Planeta Vulcânico', 'Planeta Oceânico', 'Planeta Florestal', 'Planeta Gelado'])
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Classes')
plt.show()
"""