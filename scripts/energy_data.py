import os
import glob
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
import studio2021

path = os.path.dirname(__file__)
print(path)
filename = ('1k_results.csv')
filepath = os.path.join("/msdc-thesis/results", filename)
dadus = pd.read_csv(filepath)

# Selects % of the data
dadus = dadus.sample(frac=0.1, random_state=0)

print(f'Number of points: {len(dadus)}')
dadus.head()

# All of the features of interest
selected_inputs = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated'
]

# Compute the square and sqrt of each feature
all_features = []
for data_input in selected_inputs:
    square_feat = data_input + '_square'
    sqrt_feat = data_input + '_sqrt'

    # TODO compute the square and square root as two new features
    dadus[square_feat] = dadus[data_input]**2
    dadus[sqrt_feat] = np.sqrt(dadus[data_input])
    all_features.extend([data_input, square_feat, sqrt_feat])


# Split the data into features and energy
energy = dadus['energy']
dadus = dadus[all_features]

dadus.head()

# split data
train_and_validation_dadus, test_dadus, train_and_validation_energy, test_energy = \
    train_test_split(dadus, energy, test_size=.20, random_state=6)
train_dadus, validation_dadus, train_energy, validation_energy = \
    train_test_split(train_and_validation_dadus,
                     train_and_validation_energy, test_size=.125, random_state=6)

# standardization
train_scaler = StandardScaler().fit(train_dadus)

train_dadus = train_scaler.transform(train_dadus)
validation_dadus = train_scaler.transform(validation_dadus)
test_dadus = train_scaler.transform(test_dadus)

# Linear Regression
linear_model = LinearRegression().fit(train_dadus, train_energy)
predict_energy = linear_model.predict(test_dadus)
test_rmse = sqrt(mse(predict_energy, test_energy))

# Ridge Regression
l2_penalty = np.logspace(-5, 5, 11, base=10)
data = []
for l2 in l2_penalty:
    ridge_model = Ridge(l2, random_state=0).fit(train_dadus, train_energy)
    train_predict_energy = ridge_model.predict(train_dadus)
    train_rmse = sqrt(mse(train_predict_energy, train_energy))
    validation_predict_energy = ridge_model.predict(validation_dadus)
    validation_rmse = sqrt(mse(validation_predict_energy, validation_energy))
    data.append({
        'l2_penalty': l2,
        'model': ridge_model,
        'train_rmse': train_rmse,
        'validation_rmse': validation_rmse
    })
ridge_data = pd.DataFrame(data)

# Plot the validation RMSE as a blue line with dots
plt.plot(ridge_data['l2_penalty'], ridge_data['validation_rmse'],
         'b-^', label='Validation')
# Plot the train RMSE as a red line dots
plt.plot(ridge_data['l2_penalty'], ridge_data['train_rmse'],
         'r-o', label='Train')

# Make the x-axis log scale for readability
plt.xscale('log')

# Label the axes and make a legend
plt.xlabel('l2_penalty')
plt.ylabel('RMSE')
plt.legend()

print(ridge_data)


def print_coefficients(model, features):
    """
    This function takes in a model column and a features column. 
    And prints the coefficient along with its feature name.
    """
    feats = list(zip(features, model.coef_))
    print(*feats, sep="\n")


index = ridge_data['validation_rmse'].idxmin()
best_row = ridge_data.loc[index]
best_l2 = best_row['validation_rmse']

ridge_predict = best_row['model'].predict(test_dadus)
test_rmse = sqrt(mse(ridge_predict, test_energy))
print(test_rmse)

print_coefficients(best_row['model'], all_features)
num_zero_coeffs_ridge = 0


# LASSO
l1_penalty = np.logspace(1, 7, 7, base=10)
data = []
for l1 in l1_penalty:
    lasso_model = Lasso(l1, random_state=0).fit(train_dadus, train_energy)
    train_predict_energy = lasso_model.predict(train_dadus)
    train_rmse = sqrt(mse(train_predict_energy, train_energy))
    validation_predict_energy = lasso_model.predict(validation_dadus)
    validation_rmse = sqrt(mse(validation_predict_energy, validation_energy))
    data.append({
        'l1_penalty': l1,
        'model': lasso_model,
        'train_rmse': train_rmse,
        'validation_rmse': validation_rmse
    })
lasso_data = pd.DataFrame(data)

plt.plot(lasso_data['l1_penalty'], lasso_data['validation_rmse'],
         'b-^', label='Validation')

# Plot the train RMSE as a red line dots
plt.plot(lasso_data['l1_penalty'], lasso_data['train_rmse'],
         'r-o', label='Train')

# Make the x-axis log scale for readability
plt.xscale('log')

# Label the axes and make a legend
plt.xlabel('l1_penalty')
plt.ylabel('RMSE')
plt.legend()

# Coefficient Inspection
index = lasso_data['validation_rmse'].idxmin()
best_row = lasso_data.loc[index]
best_l1 = best_row['validation_rmse']

lasso_predict = best_row['model'].predict(test_dadus)
test_rmse = sqrt(mse(lasso_predict, test_energy))
print(test_rmse)

print_coefficients(best_row['model'], all_features)
num_zero_coeffs_lasso = 28

for feature, coef in zip(all_features, best_row['model'].coef_):
    if abs(coef) <= 10 ** -17:
        print(feature)
