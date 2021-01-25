### Tuning a CART's hyperparameters
### machine learning model
### parameters: learned from data; CART example: split-point of a node, split feature of a node
### hyperparameters: not learned from data, set prior to training; max_depth, min_samples_leaf,
### splitting criterion
###
### what is hyperparameter tunintg
### Problem: search for a set of optimal hyperparameters
### Solution: find a set of optimal hyperparameters that results in an optimal model
### Optimal model: yields an optimal score
### Score: in sklearn defaults to accuracy (classification) and R^2
### Cross validation is used to estimate the generalization error
### Why tune hyperparameters?
### Approaches to hyperparameter tuning
### Grid search, Random search, Bayesian optimization, Genetic Algorithms


### Grid search cross validation
### manually set a grid of discrete hyperparameter values
### set a metric for scoring model performance
### search exhausitively through the grid
### for each set of hyperparameters
### hyperparameters grids: max_depth = {2,3,4}
### min_samples_leaf = {0.05, 0.1}
### CV scores
### Inspecting the hyperparameters of a CART in sklearn
### from sklearn.tree import DecisionTreeClassifier

### print(dt.get_params())
### Tuning a CART's hyperparameters
###
from sklearn.model_selection import GridSearchCV
###
grid_dt.fit(X_train, y_train)

###
# grid_dt.best_params_

####
print('Best hyperparameters:\n', best_hyperparameters)

####
grid_dt.best_score_

####
grid_dt.best_estimator_

#### Extracting the best esimator
####

# Define params_dt
params_dt = {'max_depth' : [2,3,4], 'min_samples_leaf' : [0.12, 0.14, 0.16, 0.18]}


#
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)


#
# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))



# Tuning an RF's hyperparameters
# Random Forests hyperparameters
# CART hyperparameters
# number of estimators
# bootstrap

# Inspecting RF hyperparameters in sklearn
rf.get_params()

#
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

#
params_rf = { 'n_estimators': [300, 400, 500],
              'max_depth': [4,6,8],
              'min_samples_leaf': [0.1, 0.2],
              'max_features': ['log2', 'sqrt']}

# Instantiate grid_rf
#
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, cv=3,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1)

#
grid_rf.fit()

# extract
grid_rf.best_params_

#
grid_rf.best_estimator_

#
# Define the dictionary 'params_rf'
params_rf = {'n_estimators': [100, 350, 500],
'max_features': ['log2', 'auto', 'sqrt'],
'min_samples_leaf': [2, 10, 30]}

#
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)


#
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred)**0.5

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test))

#### How far you have come
#### Chapter1: Decision tree learning
#### Chapter2: Generalization error, cross validation, ensembling
#### Chapter3: Bagging and Random Forests
#### Chapter4: AdaBoost and Gradient-Boosting
#### Chapter5: Model Tuning

####