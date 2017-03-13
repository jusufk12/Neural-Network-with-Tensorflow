import tensorflow as tf
import pandas as pa
import numpy as np
from sklearn import cross_validation

tf.reset_default_graph()

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
hm_epoch = 200
batch_size = 50
display_step = 1
logs_path = "/tmp/mnist/1"

#inputs
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, shape = [None, 4], name="x-input") # 4 featrues 
	y = tf.placeholder(tf.float32, shape = [None, n_classes], name="y-input") # 3 classes 
	tf.histogram_summary("input-x", x)
	tf.histogram_summary("input-y", y)
#weights
with tf.name_scope('weights'):
	W = tf.Variable( tf.zeros( [ n_features, n_classes]))
	tf.histogram_summary("weights", W)
	
#biases
with tf.name_scope('biases'):
	b = tf.Variable( tf.zeros( [n_classes ] ) )
	tf.histogram_summary("biases", b)
	
def train_neural_network(x):
	with tf.name_scope('softmax'):
		prediction = tf.nn.softmax( tf.matmul( x, W ) + b )
		
	with tf.name_scope('loss'):	
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y)) #loss
		#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
		#With Gradient Descent result is a bit lower

	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	# create a summary for our cost and accuracy
	tf.scalar_summary("cost", cross_entropy)
	tf.scalar_summary("accuracy", accuracy)
	summary_op = tf.merge_all_summaries()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())	

		writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
		for epoch in range(hm_epoch):
			avg_cost = 0.	
			epoch_loss = 0
			total_batch = int(total_len/batch_size)
			for i in range(total_batch-1):
				batch_x = X_train[i*batch_size:(i+1)*batch_size]
				batch_y = Y_train[i*batch_size:(i+1)*batch_size]
				_, c, p, summary = sess.run([optimizer, cross_entropy, prediction, summary_op], feed_dict={x: batch_x, y:batch_y})
				writer.add_summary(summary, epoch)
				avg_cost += c / total_batch
				
				print( '\r', i, ':', sess.run( accuracy, feed_dict = { x: batch_x, y: batch_y } ), end = '' )

			label_value = batch_y
			estimate = p
			err = label_value-estimate
			print ("num batch:", total_batch)
			# Display logs per epoch step
			if epoch % display_step == 0:
				print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
				print ("[*]----------------------------")
				for i in range(3):
					print ("label value:", label_value[i], "estimated value:", estimate[i])
				print ("[*]============================")
		print ("Optimization Finished!")
		print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
train_neural_network(x)


#tensorboard --logdir=run1:/tmp/mnist/1 --port=6006