import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#%matplotlib inline


num_vectors = 1000
num_clusters = 3
num_steps = 1000
from sklearn import cluster, datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

Y = pd.DataFrame(iris.target)
Y.columns = ['Targets']
plt.figure(figsize = (14,7))

colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 2, 1)
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[Y.Targets], s = 40)
plt.title('Sepal')

plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s = 40)
plt.title('Petal')

plt.show()

vectors = tf.constant(X_iris)
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[num_clusters,-1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

print (expanded_vectors.get_shape())
print (expanded_centroids.get_shape())

distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),
	reduction_indices=[1])for c in range(num_clusters)])

update_centroids = tf.assign(centroids, means)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

print("centoroids")
print(centroid_values)
print(y_iris)
print(assignment_values)

data = {"x": [], "y": [], "z": [], "w": [], "cluster": []}
for i in range(len(assignment_values)):
    data["x"].append(X_iris[i][0])
    data["y"].append(X_iris[i][1])
    data["z"].append(X_iris[i][2])
    data["w"].append(X_iris[i][3])
    data["cluster"].append(assignment_values[i])
df = pd.DataFrame(data)

sns.lmplot("x", "y", data=df, 
           fit_reg=False, size=7, 
           hue="cluster", legend=False)
plt.show()


sns.lmplot("z", "w", data=df, 
           fit_reg=False, size=7, 
           hue="cluster", legend=False)
plt.show()

#raw data with t-sne and pca
X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)
pca = PCA(n_components=2)
X_pca = pca.fit(iris.data).transform(iris.data)
#X_pca = PCA().fit_transform(iris.data)

#clustered data with t-sne and pca
X_tsne1 = TSNE(learning_rate=100).fit_transform(df)
pca1 = PCA(n_components=2)
X_pca1 = pca1.fit(df).transform(df)


plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=colormap[Y.Targets])
plt.subplot(1,2,2)
plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1],c=colormap[Y.Targets])
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colormap[Y.Targets])
plt.subplot(1,2,2)
plt.scatter(X_pca1[:, 0], X_pca1[:, 1], c=colormap[Y.Targets])
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1],c=colormap[Y.Targets])
plt.subplot(1,2,2)
plt.scatter(X_pca1[:, 0], X_pca1[:, 1], c=colormap[Y.Targets])
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=colormap[Y.Targets])
plt.subplot(1,2,2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colormap[Y.Targets])
plt.show()

