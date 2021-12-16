from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy import sqrt

carbon = pd.read_csv('assemblies_data.csv') 

# Selects 1% of the data
carbon = carbon.sample(frac=0.01, random_state=0)

print(f'Number of points: {len(carbon)}')
carbon.head()

# # All of the features of interest
# selected_inputs = [
#     'bedrooms', 
#     'bathrooms',
#     'sqft_living', 
#     'sqft_lot', 
#     'floors', 
#     'waterfront', 
#     'view', 
#     'condition', 
#     'grade',
#     'sqft_above',
#     'sqft_basement',
#     'yr_built', 
#     'yr_renovated'
# ]

# # Compute the square and sqrt of each feature
# all_features = []
# for data_input in selected_inputs:
#     square_feat = data_input + '_square'
#     sqrt_feat = data_input + '_sqrt'
    
#     # TODO compute the square and square root as two new features
#     sales[square_feat] = sales[data_input]**2
#     sales[sqrt_feat] = np.sqrt(sales[data_input])
#     all_features.extend([data_input, square_feat, sqrt_feat])


# # Split the data into features and price
# price = sales['price']
# sales = sales[all_features]

# sales.head()

# # split data
# train_and_validation_sales, test_sales, train_and_validation_price, test_price = \
#     train_test_split(sales, price, test_size=.20, random_state=6)
# train_sales, validation_sales, train_price, validation_price = \
#     train_test_split(train_and_validation_sales,
#                      train_and_validation_price, test_size=.125, random_state=6)

# # standardization
# train_scaler = StandardScaler().fit(train_sales)

# train_sales = train_scaler.transform(train_sales)
# validation_sales = train_scaler.transform(validation_sales)
# test_sales = train_scaler.transform(test_sales)

# # Linear Regression
# linear_model = LinearRegression().fit(train_sales, train_price)
# predict_price = linear_model.predict(test_sales)
# test_rmse = sqrt(mse(predict_price, test_price))

# # Ridge Regression
# l2_penalty = np.logspace(-5, 5, 11, base=10)
# data = []
# for l2 in l2_penalty:
#     ridge_model = Ridge(l2, random_state=0).fit(train_sales, train_price)
#     train_predict_price = ridge_model.predict(train_sales)
#     train_rmse = sqrt(mse(train_predict_price, train_price))
#     validation_predict_price = ridge_model.predict(validation_sales)
#     validation_rmse = sqrt(mse(validation_predict_price, validation_price))
#     data.append({
#         'l2_penalty': l2,
#         'model': ridge_model,
#         'train_rmse': train_rmse,
#         'validation_rmse': validation_rmse
#     })
# ridge_data = pd.DataFrame(data)

# # Plot the validation RMSE as a blue line with dots
# plt.plot(ridge_data['l2_penalty'], ridge_data['validation_rmse'],
#          'b-^', label='Validation')
# # Plot the train RMSE as a red line dots
# plt.plot(ridge_data['l2_penalty'], ridge_data['train_rmse'],
#          'r-o', label='Train')

# # Make the x-axis log scale for readability
# plt.xscale('log')

# # Label the axes and make a legend
# plt.xlabel('l2_penalty')
# plt.ylabel('RMSE')
# plt.legend()

# print(ridge_data)


# def print_coefficients(model, features):
#     """
#     This function takes in a model column and a features column. 
#     And prints the coefficient along with its feature name.
#     """
#     feats = list(zip(features, model.coef_))
#     print(*feats, sep="\n")


# index = ridge_data['validation_rmse'].idxmin()
# best_row = ridge_data.loc[index]
# best_l2 = best_row['validation_rmse']

# ridge_predict = best_row['model'].predict(test_sales)
# test_rmse = sqrt(mse(ridge_predict, test_price))
# print(test_rmse)

# print_coefficients(best_row['model'], all_features)
# num_zero_coeffs_ridge = 0


# #LASSO
# l1_penalty = np.logspace(1, 7, 7, base=10)
# data = []
# for l1 in l1_penalty:
#     lasso_model = Lasso(l1, random_state=0).fit(train_sales, train_price)
#     train_predict_price = lasso_model.predict(train_sales)
#     train_rmse = sqrt(mse(train_predict_price, train_price))
#     validation_predict_price = lasso_model.predict(validation_sales)
#     validation_rmse = sqrt(mse(validation_predict_price, validation_price))
#     data.append({
#         'l1_penalty': l1,
#         'model': lasso_model,
#         'train_rmse': train_rmse,
#         'validation_rmse': validation_rmse
#     })
# lasso_data = pd.DataFrame(data)

# plt.plot(lasso_data['l1_penalty'], lasso_data['validation_rmse'],
#          'b-^', label='Validation')

# # Plot the train RMSE as a red line dots
# plt.plot(lasso_data['l1_penalty'], lasso_data['train_rmse'],
#          'r-o', label='Train')

# # Make the x-axis log scale for readability
# plt.xscale('log')

# # Label the axes and make a legend
# plt.xlabel('l1_penalty')
# plt.ylabel('RMSE')
# plt.legend()

# # Coefficient Inspection
# index = lasso_data['validation_rmse'].idxmin()
# best_row = lasso_data.loc[index]
# best_l1 = best_row['validation_rmse']

# lasso_predict = best_row['model'].predict(test_sales)
# test_rmse = sqrt(mse(lasso_predict, test_price))
# print(test_rmse)

# print_coefficients(best_row['model'], all_features)
# num_zero_coeffs_lasso = 28

# for feature, coef in zip(all_features, best_row['model'].coef_):
#     if abs(coef) <= 10 ** -17:
#         print(feature)