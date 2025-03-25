import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

model_872 = load_model("models/0.872_no_overfit.keras")

df = pd.read_csv('teste.csv')
x = df.drop('id', axis=1)
scaler = StandardScaler()
x = scaler.fit_transform(x)
pred_872 = model_872.predict(x)
pred_y_classes_872 = np.argmax(pred_872, axis=1)

df['target'] = pred_y_classes_872
df[['id', 'target']].to_csv("predicoes0.872.csv", index=False)
