import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# import data
train_df = pd.read_csv('Data\\train.csv')
x0 = train_df.iloc[:,2:]
y0 = train_df['y']
x_iter = [x0, np.square(x0), np.exp(x0), np.cos(x0), pd.Series(np.ones_like(y0))]

# loop database ampliation with model fitting and calculate at every iteration the test and train errors
models = []
RMSE_val_model = []
RMSE_train_model = []

for iter in range(len(x_iter)):
    # increment the database
    X = pd.concat(x_iter[0:iter+1], axis = 1, ignore_index=True)
    # create a new linear regression model
    models.append(LinearRegression(normalize=True))
    # generate split for cross validation
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    # fit models and get prediction errors on test and validation set
    RMSE_val_split = []
    RMSE_train_split = []

    # perform 10-fold cross validation
    for train_index, val_index in kf.split(X):
        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y0.iloc[train_index], y0.iloc[val_index]

        # fit model
        models[iter].fit(x_train, y_train)

        # get predictions
        y_val_pred = models[iter].predict(x_val)
        y_train_pred = models[iter].predict(x_train)

        # get mse on predictions
        RMSE_val_split.append(mean_squared_error(y_val, y_val_pred) ** 0.5)
        RMSE_train_split.append(mean_squared_error(y_train, y_train_pred) ** 0.5)

    # get mean prediction error for i-th model
    RMSE_val_model.append(np.mean(RMSE_val_split))
    RMSE_train_model.append(np.mean(RMSE_train_split))

# plt the errors
# Create plots with pre-defined labels.
plt.plot(RMSE_val_model, 'k--', label='Average Validation Error')
plt.plot(RMSE_train_model, 'k:', label='Average Test Error')
legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()
