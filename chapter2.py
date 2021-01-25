########### chapter2
###########


############ supervised learning -- under the hood
############ goals of supervised learning
############ find a model that best approximates f
############ end goal: fhat should achieve a low predictive error on unseen datasets
############ Difficulties in approximating f
############ overfitting, fhat fits the training set noise
############ underfitting, fhat is not flexible enough to approximate f
############




############# Overfitting
############# Undefitting mental abstraction level that enables him to understand the calculus
############# Generalization error of fhat: does fhat generalize well on unseen data?
############# bias: error term that tells you, on average, how much fhat != f
############# variance: tells you how much fhat is incosistent over different training sets
############# high variance models leads to overfitting?


############# Model complexity: sets the flexibility of model

############# generalization error of fhat = bias^2 + variance + irreducible error
############# find the model complexity that gives the smallest generalization error
############# low variance: precise
############# high variance: not precise

############# Estimating the Generalization error
############# cannot be done directly because
############# 1, f is unknonw
############# 2. usually you only have one dataset
############# 3, noise is unpredictable

# estimating the generalization error
# split the original dataset into test and training
# generalization error of fhat ~ test set error of fhat
# Better model evaluation with cross-validation
# test set should not be touched until we are confident about fhat's performance
# evaluating fhat on training set: biased estimate, fhat has already seen all training points
# solution --> cross validation (CV)
# K-fold CV
# Hold-out CV
# CV error = mean of 10 errors
# Diagnose Variance Problems
# If fhat suffers from high variance: CV error of fhat > training set error of fhat
# decrease max depth, increase min samples per leaf
#

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

SEED = 123

X_train, X_test, y_train, y_test = train_test_split((X, y, test_size = 0.3, random_state = SEED))

##
MSE_CV = - cross_val_score(dt, X_train, y_train, cv = 10,
                           scoring = 'neg_mean_squared_error',
                           n_jobs = -1)

##
# Diagnose bias and variance problems

##

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)


# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10,
                       scoring='neg_mean_squared_error',
                       n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = ((MSE_CV_scores).mean())**(0.5)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))


# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(0.5)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))


## Ensemble learning
## Advantages of CARTs
## simple to understand
## simple to interpret
## easy to use
## flexibility
## preprocessing
## Limitation of CARTs

## can only produce orthogonal decision boundaries
## sensitive to small variations in the training set
## high variance, unconstrained CARTs may overfit the training set


## Ensemble learning: Meta-model
## final prediction: more robust and less prone to errors
## Best results: models are skillful in different ways
## Ensemble learning: A visual explanation
## Ensemble learning in practice: voting classifier
## Binary classification task
## N classifiers make predictions

## voting classifier in sklearn
##
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#
from sklearn.ensemble import VotingClassifier

#
SEED = 1

#
classifiers = [ ('Logistic Regression', lr),
                ('K Nearest Neighbours', knn),
                ('Classification Tree', dt)]

#
for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    # predict
    y_pred = clf.predict(X_test)
    #
    print()

###
### Instantiate a VotingClassifier 'vc'
### VotingClassifier(estimators = classifiers)
### vc.fit()
###


# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))


#
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))