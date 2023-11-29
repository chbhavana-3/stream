import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
                    'age':np.random.randint(1,40,1000), 
                   'exp':np.random.randint(1,20,1000),
                   'salary':np.random.randint(1,30,1000),
                    
                 })

df['salary'] = df['age']*6 + df['exp']*3

X = df.drop(['salary'],axis=1)
y = df['salary']

import streamlit as st
a = st.radio("select features you want to select", [item for item in X.columns])
st.write(f"selected feature is {a}")
# print(a)

X = X.drop([a], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression().fit(X_train,y_train)
y_pred_train = model.predict(X_train)
score_train = r2_score(y_train, y_pred_train)
print(f"\n R-square for train is {score_train*100}\n")
st.write(f"\n R-square for train is {score_train*100}\n")

y_pred_test = model.predict(X_test)
score_test = r2_score(y_test, y_pred_test)
print(f"\n R-square for test is {score_test*100}\n")
st.write(f"\n R-square for test is {score_test*100}\n")

decision_tree_regressor = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
y_pred_tree = decision_tree_regressor.predict(X_train)
mse_tree = mean_squared_error(y_train, y_pred_tree)
print(f'Mean Squared Error (Decision Tree)for train is {mse_tree}\n')
st.write(f'\n Mean Squared Error (Decision Tree)for train is {mse_tree}\n')

decision_tree_regressor = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
y_pred_tree = decision_tree_regressor.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f'Mean Squared Error (Decision Tree) {mse_tree}\n')
st.write(f'\n Mean Squared Error (Decision Tree) for test is {mse_tree}\n')

knn_regressor = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
y_pred = knn_regressor.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
print(f'Mean Squared Error(KNN): {mse}')
st.write(f"\n (f'Mean Squared Error(KNN)for train is: {mse}\n")

knn_regressor = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
y_pred = knn_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error(KNN): {mse}')
st.write(f"\n (f'Mean Squared Error(KNN) for test is: {mse}\n")