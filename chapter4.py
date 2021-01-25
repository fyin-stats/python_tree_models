############ Chapter 4
############ AdaBoost
############


############
# Boosting: Ensemble method combining several weak learners to a strong learner
# Weak learner: model doing slightly better than random guessing
# Eamples of weak learner: Decision stump

#############
# AdaBoost: Adaptive boosting
# Gradient Boosting

#############
# achieved by changing the weights of training instances
# each predictor is assigned a coefficient alpha

# alpha depends on the predictors's training error

# AdaBoost
# alter the weights of incorrectly predicted instances
# Learning rate: eta
# shrinks the alpha during the training

# Classification: Weighted majority voting
# Regression: Weighted average

# individual learner can be arbitrary
# AdaBoost classification in sklearn

#
dt = DecisionTreeClassifier(max_depth = 1, random_state = SEED)

#
adb_clf = AdaBoostClassifier(base_estimator = dt, n_estimators = 100)

# AdaBoostClassifier()

#
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)


#
# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]

#
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))



# Gradient Boosting (GB)
# sequential correction of predecessors' errors
# Does not tweak the weight sof training instances
# Fit each predictors is trained using its predecessor's residual errors as labesl
# Gradient Boosted Trees: a CART is used as a base learner
# fit to the residuals
# Shrinkage, at each step, shrink the error from last step
# shrinkage factor: eta
# Gradient boosted trees: Prediction
# Regression: GradientBoostingRegressor
# Classification: GradientBoostingClassifier
# Gradient Boosting in sklearn (auto dataset)
#
from sklearn.ensemble import GradientBoostingRegressor

# GradientBoostingRegressor()

# gbt.fit(X_train, y_train)
#

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4,
            n_estimators=200,
            random_state=2)

#
# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)


# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test**0.5

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))



# stochastic gradient boosting
# GB involves an exhaustive search procedure
# Each CART is trained to find the best split points and features
# May lead to CARTs using the same split points and maybe the same features
# Each tree is trained on a random subset of rows of the training data
# The sampled instances (40%-80% of the training set) are sampled without replacement
# Features are sampled when choosing split points
# Result: further ensemble diversity
# Effect: adding further variance to the ensemble of trees

# residual errors (computed on all data)
# (X, eta r1)
#


# stochastic gradient boosting in sklearn (auto dataset)
#
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

#
# GradientBoostingRegressor(max_depth = 1, subsample = 0.8, max_features = 0.2, n_estimatorws = 300, random_state = SEED)
# sgbt.fit
# sgbt.predict

#
#
# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)