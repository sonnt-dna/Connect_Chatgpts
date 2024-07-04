# Thủc GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv_=3, njobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Kổt vất nhệ dựng trường GridSearchCV
gib_model = grid_search.best_estimator_
gred_param = grid_search.best_params

print(f'Best Parameters: ${ grid_param }')
