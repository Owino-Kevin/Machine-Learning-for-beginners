# In this tutorial, we will learn how to carry out machine learning in Python.
# Linear Regression
import matplotlib.pyplot as plt

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]  # These are arrays

plt.scatter(x, y)
# plt.show()
from scipy import stats

slope, intercept, r, p, std_err = stats.linregress(x,
                                                   y)  # A method that returns important key values in linear regression


def myfunc(
        x):  # A function that uses slope and intercept value to return a new value representing where on the y-axis the corresponding x value is placed
    return slope * x + intercept


mymodl = list(map(myfunc,x))  # Runs each value of the x array through the function to give new array with new values for the y-axis.

plt.scatter(x, y)  # Draw the original scatter plot
plt.plot(x, mymodl)  # Draw the linear regression
# plt.show()  # Display the diagram
print(r)  # r is the coefficient of correlation. It shows the relationship between x and y. Since it is -0.76, there is a relationship.

# Therefore we can use linear regression for forecasting. Now, given that x = 10, forecast the value of y at that point.
y = myfunc(10)
print(y)

# The Polynomial Distribution: Used when data does not fit the normal distribution.
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
mymodel = np.poly1d(np.polyfit(x, y, 3))  # The polynomial model
myline = np.linspace(1, 22, 100)  # Specify how the line displays, start from 1 and end at 22
plt.scatter(x, y)  # Scatter plot
plt.plot(myline, mymodel(myline))  # Draw the line of polynomial regression
# plt.show()

# Goodness of fit in this polynomial model.
from sklearn.metrics import r2_score

print(r2_score(y, mymodel(
    x)))  # 0.94 shows there is a very good relationship, hence this model can be used for prediction.

# Prediction: Predict y when x is 17
y = mymodel(17)
print(y)  # Y is 88.87

# Good fit? Yes.

# Multiple Regression: More than one independent value.
import pandas

df = pandas.read_csv("C:/Users/VICTOR ORWA/Downloads/data (1).csv")
print(df)
X = df[['Weight', 'Volume']]  # Declaring the independent variables
y = df['CO2']  # Declaring the dependent variables
from sklearn import linear_model  # Import the required packages
regr = linear_model.LinearRegression()  # The regression model
regr.fit(X, y)  # Fitted regression model
predictedCO2 = regr.predict([[2300, 1300]])  # Predicting the CO2 emitted by 2300 tonne car with 1300 engine size
print(predictedCO2)  # 107.208 volume of CO2 is emitted.
print(regr.coef_)  # To print the regression coefficients.

# Train/Test: To find out if the model is good enough.
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x
train_x = x[:80]  # 80% for the training set
train_y = y[:80]
test_x = x[80:]  # 20% for the testing set
test_y = y[80:]


plt.scatter(train_x, train_y)
# plt.show()
plt.scatter(test_x, test_y)
# plt.show()

# Fitting the data;
mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))
myline = np.linspace(0, 6, 35)
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
# plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(train_y, mymodel(train_x))  # R-squared measures the relationship between x and y.
print(r2) # A 0.79 proves a good relationship.

# Now we bring in the testing Set.
r2 = r2_score(test_y, mymodel(test_x))
print(r2)  # A 0.809 shows that the model fits the testing set. We are now confident that we can use the model for prediction.

# Since our model is OK, we can now predict.
print(mymodel(5))  # Corresponding value of y is 22.88

# Python Decision Tree.
import pandas
df = pandas.read_csv("C:/Users/VICTOR ORWA/Desktop/Python Data Analysis/Machine Learning/D.Tree.csv")
print(df)
# But to make the decision tree, all the data has to be numerical. So we convert Nationality and Go to numerical values.
d = {'UK': 0, 'USA': 1, 'N': 2}
df["Nationality"] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df["Go"] = df['Go'].map(d)
print(df)

# We then seperate the feature column from target column.
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df["Go"]
print(X)
print(y)

# Let us create the Decision Tree.
import pandas
from sklearn.externals import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
# tree.plot_tree(dtree, feature_names=features)
# plt.show()
print(dtree.predict([[40, 10, 7, 1]]))  # Result of 1 means Go.

# Confusion Matrix: A table used in classification problems to asses where errors in the model were made.
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1, 0.9, size = 1000)
predicted = numpy.random.binomial(1, 0.9, size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix= confusion_matrix, display_labels=[False, True])

# cm_display.plot()
# plt.show()
Accuracy = metrics.accuracy_score(actual, predicted)
print(Accuracy)  # Accuracy is 0.834

Precision = metrics.precision_score(actual, predicted)
print(Precision)  # Of the positives(True positives and False Positives) predicted 0.903 are truly positive.

# Sensitivity: Measures ow good the model is at predicting positives. It's better than Precision.
Sensitivity = metrics.recall_score(actual, predicted)
print(Sensitivity)  # Of ALL the positives(True positives and False Negatives), 0.915 are predicted as positives.

# Specificity: How well the model is at predicting Negative results.
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
print(Specificity)  # 6.32%

# F-score: It's a harmonic mean of precision and sensitivity
F1_score = metrics.f1_score(actual, predicted)
print(F1_score)  # The harmonic mean is 0.909