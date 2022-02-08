import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# path = os.path.abspath('')
# filename = ('1k_results.csv')
# filepath = os.path.join(path, "results", filename)
# print(filepath)

data = pd.read_csv('/Users/preston/Documents/GitHub/msdc-thesis/tool/results/1k_results.csv')

labels_drop = ['filename', 'num_adiabatic', 'setback', 'area_buildable',
               'carbon', 'surf_area', 'volume', 'surf_vol_ratio', 'kgCO2e']
data.drop(labels=labels_drop, axis=1, inplace=True)
print(data.head())
# data.isnull().sum()

X, y = data.iloc[:,:-1], data.iloc[:,-1]

# train xgboost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = xgb.XGBRegressor()
params = {
    "max_depth": np.arange(3, 50, 1), 
    "gamma": np.arange(0, 20, 1), 
    "lambda": np.arange(0, 1, .1)
    }
# params['eval_metric'] = 'mae'

randomized_search = RandomizedSearchCV(model, params, n_iter=20, 
                                       scoring="neg_mean_absolute_error", cv=3)
                                       #scoring='neg_mean_squared_error'
randomized_search.fit(X_train, y_train)

# examine error of optimized model
xgboost_reg = randomized_search.best_estimator_
preds = xgboost_reg.predict(X_test)

# rmse loss
# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE: %f" % (rmse))

# mae loss
mae = mean_absolute_error(y_test, preds)
print("RMSE: %f" % (mae))

pickle_out = open('xgboost_reg.pkl','wb')
pickle.dump(xgboost_reg, pickle_out)
pickle_out.close()
print('Model has been pickled')