import tensorflow as tf
import pandas as pa
import numpy as np
from sklearn import cross_validation
import sklearn as sk

dfrm = pa.read_csv('iris.csv')

data = dfrm.to_csv( header = False, index = False ).split('\n')

inputs = np.asarray( list ( map ( lambda x : [ float(y) for y in x.split(',')[:-1] ], data[:-1] ) ) )
class_item = [ y for y in set( map ( lambda x : x.split(',')[-1], data[:-1] ) ) ]

num_of_class    = len( class_item )
num_of_data     = len( inputs )
num_of_feature  = len( inputs[0] )

class_map     = { class_item[i] : i for i in range( num_of_class ) }
class_inv_map = { i : class_item[i] for i in range( num_of_class ) }

cls_tmp = np.asarray( list ( map ( lambda x : class_map.get( x.split(',')[-1] ), data[:-1] ) ) )
targets = np.asarray( list ( map ( lambda x : [ int( x == i ) for i in range( num_of_class ) ], cls_tmp ) ) )


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
inputs, targets, test_size=0.3, random_state=42)
#total_len = tf.shape(X_train)[0]
total_len = X_train.shape[0]


n_classes = 3 # U ovom slucaju tri, 1-> Iris-setosa, Iris-versicolo, Iris-virginica
n_features = 4
learning_rate = 0.01
hm_epoch1 = 30
hm_epoch2 = 70
batch_size = 50
display_step = 1

#inputs
x = tf.placeholder(tf.float32, shape = [None, 4]) # 4 featrues 
#outputs
y = tf.placeholder(tf.float32, shape = [None, n_classes]) # 3 classes 
#weights
W = tf.Variable( tf.zeros( [ n_features, n_classes]))
#biases
b = tf.Variable( tf.zeros( [n_classes ] ) )


prediction = tf.nn.softmax( tf.matmul( x, W ) + b )
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y)) #loss
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#With Gradient Descent result is a bit lower
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()
saver = tf.train.Saver()
print("Starting 1st session...")
with tf.Session() as sess:
	sess.run(init)			
	for epoch in range(hm_epoch1):
		avg_cost = 0.	
		epoch_loss = 0
		total_batch = int(total_len/batch_size)
		for i in range(total_batch-1):
			batch_x = X_train[i*batch_size:(i+1)*batch_size]
			batch_y = Y_train[i*batch_size:(i+1)*batch_size]
			_, c, p = sess.run([optimizer, cross_entropy, prediction], feed_dict={x: batch_x, y:batch_y})

			avg_cost += c / total_batch
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		print( '\rNumber of epoch:', epoch, 'Training accuracy:', sess.run( accuracy, feed_dict = { x: batch_x, y: batch_y } ), end = '' )

	print ("\nOptimization Finished!")
	print ("Testing Accuracy after first session:", accuracy.eval({x: X_test, y: Y_test}))
	save_path = saver.save(sess, "model.ckpt")
	print("Model saved in file: %s" % save_path)


print('\n')
print("Starting 2nd session...")
with tf.Session() as sess:
	sess.run(init)	
	saver.restore(sess, "model.ckpt")		
	for epoch in range(hm_epoch2):
		avg_cost = 0.	
		epoch_loss = 0
		total_batch = int(total_len/batch_size)
		for i in range(total_batch-1):
			batch_x = X_train[i*batch_size:(i+1)*batch_size]
			batch_y = Y_train[i*batch_size:(i+1)*batch_size]
			_, c, p = sess.run([optimizer, cross_entropy, prediction], feed_dict={x: batch_x, y:batch_y})

			avg_cost += c / total_batch
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		print( '\rNumber of epoch: ', epoch, 'Training accuracy:', sess.run( accuracy, feed_dict = { x: batch_x, y: batch_y } ), end = '' )

	print ("\nSecond Optimization Finished!")
	print ("Number of batchs:", total_batch)
	print ("Testing Accuracy after continuing second session:", accuracy.eval({x: X_test, y: Y_test}))
	

	y_p = tf.argmax(prediction, 1)
	val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: X_test, y: Y_test})
	y_true = np.argmax(Y_test, 1)
	#print ("Precision", sk.metrics.precision_score(y_true, y_pred))
	#print ("Recall", sk.metrics.recall_score(y_true, y_pred))
	#print ("f1_score", sk.metrics.f1_score(y_true, y_pred))
	print ("confusion_matrix")
	print (sk.metrics.confusion_matrix(y_true, y_pred))
