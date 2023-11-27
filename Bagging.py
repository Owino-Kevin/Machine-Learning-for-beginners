# Bootstrap Aggregation(Bagging): Resolves overfitting for classification or regression problems like in Decision Trees.
# The aim is to improve accuracy and performance of machine learning algorithms.
import matplotlib.pyplot as plt
# Let us Evaluate a Base Classifier for Decision Trees.
# We shall use classes of wines in the sklearn dataset.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Let us load the data into x(input feature) and y(target feature). as_frame is used to maintain features names.
data = datasets.load_wine(as_frame=True)

X = data.data
y = data.target

# Let us now split our data to train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Let us now instantiate base classifier and fit into the training data.
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred = dtree.predict(X_train)))  # Accuracy of 1.0
print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred = y_pred))  # Accuracy of 82%
# The base classifier achieves an accuracy of 82% which is good.

# Now we can see how the Bagging Classifier outperforms a single Decision Tree Classifier.
from sklearn.ensemble import BaggingClassifier

# We create a range of values that represents the number of estimators we want to use in each ensemble.
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]

# We create a loop, storing the models and scores in seperate lists for later visualizations.
# This is becaise we need to see how Bagging classifier performs with differing values of n_estimators.
models = []
scores = []

for n_estimators in estimator_range:
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)  # Create a Bagging Classifier
    clf.fit(X_train, y_train)  # Fit the model
    models.append(clf)  # Appends the model and scores to their respective list
    scores.append(accuracy_score(y_true=y_test, y_pred = clf.predict(X_test)))

# With the models and scores stored, we can now visualize the improvement in model performance.
# First, we generate the plot of scores against the number of estimators.
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

# Adjust labels and font for visibility.
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize the plot.
plt.show()  # From the graph, we see that the score goes up to 95.5% which is 13% better than the base classifier.

# Finaly, let us now generate the decision tree from the Baseline Classifier.
from sklearn.tree import plot_tree
clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)

clf.fit(X_train, y_train)
plt.figure(figsize=(10,10))
plot_tree(clf.estimators_[0], feature_names= X.columns)
plt.show()  ## This shows the decision tree from Bagging Classifier.