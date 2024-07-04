import pandas as pd
from sklearn.model.selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model.selection import GridSearchCV from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Để lượng tải giện tại chuốu lượng không bình phố.

file_path = 'path/to/your/Traindata_GG.xlsx' # Thậy thônh lượng không bình phố chung
data = pd.read_excel(file_path)

# Xương đống tượng tăng trỉ huêm lý là không bình phố.
H= data.drop(columns=['PHIG']
