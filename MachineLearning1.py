# HIERARCHICAL CLUSTERING: Unsupervised ML algorithm that builds clusters by measuring dissimilarities between data.
import numpy
import matplotlib.pyplot as plt

p = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
q = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(p, q)
plt.show()

# To link the data in the scatter plot(Ward Linkage), we use Euclidean distance, and visualize using dendrogram.
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
data = list(zip(p, q))  # Turns the data into set of points.
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()

# Agglomerative Clustering;
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(p, q, c = labels)
plt.show()

# LOGISTIC REGRESSION: Solves classification problems through predicting categorical outcomes.
# X represents the size of a tumor in centimeters
import numpy
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
# Note that X has been reshaped into a column from a row for the LogidticRegression() function to work.
# y represents whether or not the tumor is cancerous(0 for No, 1 for Yes)

y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
from sklearn import linear_model

logr = linear_model.LogisticRegression()
logr.fit(X, y)


# Predict if tumor is cancerous where the size is 3.46mm
predicted = logr.predict(numpy.array([3.46]).reshape(-1, 1))
print(predicted)  # Gives a 0 which shows that tumor of that size is not cancerous

coefficient = logr.coef_
odds = numpy.exp(coefficient)
print(odds)  # Gives the coefficient of the regression as 4.035

# We can also obtain the probability that each tumor is cancerous.
def logit2prob(logr, x):
    coefficient = logr.coef_ * x + logr.intercept_  # This looks like linear regression, purpose is to obtain the intercept.
    odds = numpy.exp(coefficient)  # We take exponent of the coefficient to remove the logs.
    probability = odds/(1 + odds)  # We then convert it to probability by dividing it by 1 plus itself.
    return(probability)  # This gives the probability
print(logit2prob(logr, X))  # The probability that a tumor of the size 3.78cm is cancerous is 61%. Same for the other probabilities.

# K-MEANS: This is an unsupervised ML method for clustering data points by dividing data in to K clusters by minimizing variance in each cluster.
import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

# We then use elbow method to visualize the inertia for different values of K
from sklearn.cluster import KMeans
data = list(zip(x, y))  # Turns data into set of points/coordinates
inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 11), inertias, marker = 'o')
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()  # The elbow on the graph is at K = 2, so 2 is a good value for K. We then retrain and visualize the result.

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x, y, c = kmeans.labels_)
plt.show()
