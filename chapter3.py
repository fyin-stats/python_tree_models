#### Bagging
#### Bootstrap aggregation
#### Voting classifier
#### Uses a technique known as the bootstrap
#### Bootstrap: Sample with replacement


#### Bagging
#### Train models on different bootstrap samples
#### with the same algorithm
#### New instance, predictions


#### Classification: BaggingClassifier
#### Regression: BaggingRegressor

#### Bagging Classifier in sklearn
####
from sklearn.ensemble import BaggingClassifier

#### Instantiate a classification tree
#### Instantiate a BaggingClassifier 'bc'
#### BaggingClassifier 'bc'
#### n_jobs = -1 # uses all cores for training
####


# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)







#### Out of bag (OOG) evaluation
#### Bagging : some instances may be sampled several times for one model
#### Other instances may not be sampled at all
#### OOB instances
#### not seen by the model during training
#### can be used to evaluate the model's perofrmance (generalization error)
#### OOB evaluation
#### OOB evaluation in sklearn
####
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 1

# X_train, X_test, y_train, y_test = train_test_split()

# Instantiate a classification-tree dt
# BaggingClassifier
# accuracy_score(y+_test, y_pred)
# bc.oob_score_

# test_accuracy
# oob_accuracy
#


# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))


# Random Forests
# in bagging = base estimator, decision tree, logistic regression, neural net
# estimators use all features for training and prediction
# base estimator: Decision tree
# RF introduces further randomization in the training of individual trees
# d features are sampled at each node without replacement
# d < total number of features
# Random forests: Training
# sample d features at each split without replacement
# RandomForestClassifier
# RandomForestRegressor
# random forest achieves lower variance than individual trees\
#
rf = RandomForestRegressor(n_estimators =400, min_samples_leaf = 0.12, random_state = SEED)



#######################################
# Feature importance
# tree-based methods: enable measuring the importance of each feature in prediction
# in sklearn:
# how much the tree nodes use a particular feature (weighted average) to reduce impurity
# accessed using the attribute feature_importance_

#
import pandas as pd
import matplotlib.pyplot as plt

# create a pd.Series of feature importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

# sort importances_rf
sorted_importances_rf = importances_rf.sort_values()


# make a horizontal bar plot
sorted_importances_rf.plot(kind = 'barh', color = 'lightgreen'); plt.show()

#########################################

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
                           random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)


#
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**0.5

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


#
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()