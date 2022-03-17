import pickle
import xgboost as xgb

pickle_in = open('xgboost_reg.pkl', 'rb')
regressor = pickle.load(pickle_in)
print(type(regressor))