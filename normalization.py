import pandas as pd
df = pd.read_csv(r'C:/Users/me/Desktop/crop yeild data.csv', encoding='latin1')
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
print(scaler.fit_transform(df))
from sklearn.preprocessing import StandardScaler
scaler1=StandardScaler()
print("\n" "\n" " Z SCORE IS NOW BEING PRINTED")
print(scaler1.fit_transform(df))

