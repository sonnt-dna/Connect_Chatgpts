# đênh gia mộnh tượng kế nhân
gy_pred = best_model.predict(XTest)
mse = mean_squared_error(yTest, y_pred)
r2 = r2score(yTest, y_pred)
print('Mean Squared Error: ${MSE}')
print('R^2 Score: ${R2}')
