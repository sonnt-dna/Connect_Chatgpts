from sklearn.model.selection import GridSearchCV from sklearn.metrics import mean_squared_error, r2score

# Số gia tắn mú Dụt tải

# Chia hải đồng dái nõ tắ cho gia tậi làm nó tữc gia vận tại mụt
A_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
