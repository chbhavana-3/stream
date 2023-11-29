import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
                    'age':np.random.randint(1,30,1000), 
                   'exp':np.random.randint(1,10,1000),
                   'cg':np.random.randint(1,10,1000),
                    # 'pay':[200, 500, 750, 1200, 2000],
                    # 'pay':np.random.randint(1000,1500,1000)
                  })

# df['cg'] = df['exp']*3
df['pay'] = df['age']*6 + df['exp']*3

# print(df)
X = df.drop(['pay', 'exp'],axis=1)
y = df['pay']

import streamlit as st
# a = st.radio("Choose a feature",[item for item in X.columns])
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