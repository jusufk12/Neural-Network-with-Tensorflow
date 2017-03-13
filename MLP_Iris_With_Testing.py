import tensorflow as tf
import pandas as pa
import numpy as np


ipd = pa.read_csv("iris2.csv")
species = list(ipd['Species'].unique())
ipd['One-hot'] = ipd['Species'].map(lambda x: np.eye(len(species))[species.index(x)] )
ipd.sample(5)

shuffled = ipd.sample(frac=1)
trainingSet = shuffled[0:len(shuffled)-50]
testSet = shuffled[len(shuffled)-50:]
print(trainingSet)
print(testSet)


n_classes = 3 # U ovom slucaju tri, 1-> Iris-setosa, Iris-versicolo, Iris-virginica
n_features = 4

#inputs
x = tf.placeholder(tf.float32, shape = [None, 4]) # 4 featrues 
#outputs
y = tf.placeholder(tf.float32, shape = [None, n_classes]) # 3 classes 
#weights
W = tf.Variable( tf.zeros( [ n_features, n_classes]))
#biases
b = tf.Variable( tf.zeros( [n_classes ] ) )


def train_neural_network(x):
	prediction = tf.nn.softmax( tf.matmul( x, W ) + b )
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y)) #loss
	#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	#With Gradient Descent result is a bit lower
	optimizer = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


	#koliko puta ce ici back
	hm_epoch = 1000
	with tf.Session() as sess:
		keys = ['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width']
		sess.run(tf.initialize_all_variables())	
		for step in range(hm_epoch):	
		    trainData = trainingSet.sample(50)			
		    sess.run([optimizer, cross_entropy], feed_dict={x: [t for t in trainData[keys].values], y:[t for t in trainData['One-hot'].as_matrix()]})					
		    testiranje = sess.run( accuracy, feed_dict = {x: [t for t in trainData[keys].values], y:[t for t in trainData['One-hot'].as_matrix()] } )
		    print('\r Training accuracy-> epoch:', step, 'Accuracy:', format(testiranje), end = ' ' )
		print ('\n Testing accuracy:', sess.run(accuracy, feed_dict={x: [x for x in testSet[keys].values], 
                                    y: [x for x in testSet['One-hot'].values]}))


	
train_neural_network(x)
