import tensorflow as tf



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
from sklearn import cross_validation

train = mnist.train
test = mnist.test
train_x = train.images
train_y = train.labels
test_x = test.images
test_y = test.labels



X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
        test_x, test_y, test_size=0.3, random_state=42)
total_len = X_train.shape[0]



'''

n_classes = 10
batch_size = 128
logs_path = "/tmp/mnist/2"
'''
with tf.name_scope('inputs'):
    x = tf.placeholder('float', [None, 784])

with tf.name_scope('labels'):
    y = tf.placeholder('float')


keep_rate = 0.8
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    with tf.name_scope('prediction'):
        prediction = convolutional_neural_network(x)
    
    with tf.name_scope('loss'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    
    hm_epochs = 10
    batch_size = 50

    tf.scalar_summary("cost", cost)
    tf.scalar_summary("accuracy", accuracy)
    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            total_batch = int(total_len/batch_size)
            for i in range(total_batch):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = Y_train[i*batch_size:(i+1)*batch_size]
                _, c, summary = sess.run([optimizer, cost, summary_op], feed_dict={x: batch_x, y: batch_y})
                writer.add_summary(summary, epoch * total_batch + i)

                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print('Training accuracy:', sess.run( accuracy, feed_dict = { x: batch_x, y: batch_y } ), end = '' )
             
        print('Accuracy:',accuracy.eval({x: X_test, y: Y_test}))

train_neural_network(x)