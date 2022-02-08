import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import xgboost as xgb
<<<<<<< HEAD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
=======
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
>>>>>>> 27323b75712e589908b6a3b4898d4b5a4956a60e
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

<<<<<<< HEAD
randomized_search = RandomizedSearchCV(model, params, n_iter=20, 
                                       scoring="neg_mean_absolute_error", cv=3)
                                       #scoring='neg_mean_squared_error'
=======
parameters = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight": [1, 3, 5, 7],
     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

# parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
#               'objective': ['binary:logistic'],
#               'learning_rate': [0.05],  # so called `eta` value
#               'max_depth': [6],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7],
#               # number of trees, change it to 1000 for better results
#               'n_estimators': [5],
#               'missing': [-999],
#               'seed': [1337]}

# grid_search = GridSearchCV(model, parameters, n_jobs=5,
#                            cv=3,
#                            scoring='roc_auc',
#                            verbose=2, refit=True)
randomized_search = RandomizedSearchCV(model, parameters, n_iter=20, 
                                       scoring="neg_mean_squared_error", cv=3)
>>>>>>> 27323b75712e589908b6a3b4898d4b5a4956a60e
randomized_search.fit(X_train, y_train)
# grid_search.fit(X_train, y_train)
# examine error of optimized model
xgboost_reg = randomized_search.best_estimator_
# xgboost_reg = grid_search.best_estimator_
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

xgb.plot_importance(xgboost_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
