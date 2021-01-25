### https://learn.datacamp.com/courses/machine-learning-with-tree-based-models-in-python
### tree-based models for regression and classification
### chap1: classification
### chap2: the bias tradeoff
### chap3: bagging and random forests
### chap4: boosting
### chap5: model tuning


### classification tree: sequence of if-else questions about individual features
### objective: infer class labels
### able to capture non-linear relationships between features and labels
### don't require feature scaling

### Breast cancer dataset in 2D
### Decision-tree Diagram
### Tree diagram
### maximum depth
### classification tree in scikit learn

###
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

### Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1) # to ensure reproducibility

###
df.fit(X_train, y_train)
y_pred = df.predict(X_test)
accuracy_score(y_test, y_pred)

### Decision region: region in the feature space where all instances are assigned to one class label

### Decision Boundary: surface separating different decision regions
###


###
# Import LogisticRegression from sklearn.linear_model

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

###
# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import  LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)

###

### Building blocks of a decision  tree
### Decision tree: data structure consisting of a hierarchy of nodes
### node: question or prediction
### root: no parent node, question giving rise to two children nodes
### internal node: on parent node, question giving rise to two children nodes
### leaf: one parent node, no children nodes --> prediction
### information gain (IG): at each node, a tree asks a question
### how does the tree decide which question to ask?
### Left, nleftsamples
### Right, nrightsamples
### IG(f, sp) = I(parent) - (Nleft / N * I(left) + Nright / N I(right))
### where f denotes the feature
### sp denotes the split-point
### Classification-Tree Learning
### Nodes are grown recursively
### at each node, split the data based on
### feature f and split point sp to maximize IG(node)
### IG(node) = 0, declare the node a leaf when there is no constraint
###

### Import DecisionTreeClassifier
###
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(criterion='gini', random_state=1)

##

# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)




## Decision Tree for regression
## Auto-mpg dataset
## Regression Tree in scikit learn
##
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

## split data into 80% and 20% test
##
dt = DecisionTreeRegressor(max_depth = 4, min_samples_leaf = 0.1, random_state=3)

## Regression Tree in scikit learn

MSE(y_test, y_pred)

## Information criterion for Regression-Tree
## I(node) = MSE(node) = 1/(Nnode) sum(y_i - yhatnode)^2
## yhatnode = 1/(Nnode) * y_i


## Prediction
## yhatpred(leaf) = 1/(Nleaf) * sum y_i
##


# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)

## Evaluate the regression tree
##

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt**0.5

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))


# Predict test set labels 
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)

# Compute rmse_lr
rmse_lr = mse_lr**0.5

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))