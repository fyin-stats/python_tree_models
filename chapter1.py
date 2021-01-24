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