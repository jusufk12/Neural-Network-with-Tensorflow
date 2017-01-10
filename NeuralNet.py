import tensorflow as tf
import pandas as pa


iris = pa.read_csv("iris.csv")
'''
 Poraditi dodatno na podacima, fino ih podjeliti na dva dijela za training i testing

 U datasetu klase su napisane slovima, to treba prebaciti u brojeve, al kako sam primjetio
 zbog tenzora te klase se prebace u array pa bi klasa 1 bila [1, 0, 0], klasa d [0, 1, 0]
 i klasa 3 [0, 0, 1] taj dio takodjer treba u pred procesiranju odraditi koje jos nije uradjeno.


'''
#provjera da li je ucitalo kako treba
print(iris.head(5))

n_nodes_hl1 = 150
n_nodes_hl2 = 150


n_classes = 3 # U ovom slucaju tri, 1-> Iris_setosa, Iris-versicolo, Iris-virginica
n_features = 4 # broj inputa
batch_size = 50 # Da li ima neko optimalno rijesenje koliko uzeti?

x = tf.placeholder('float', [None, n_features]) # 4 featrues 
y = tf.placeholder(tf.int32, [None, n_classes]) # 3 classes 


def neural_network_model(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([?, n_nodes_hl1])),
					'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
					'biases': tf.Variable(tf.random_normal(n_classes))}

	#layer 1, mnozi data sa weights te dodaje biase, (input_data * weights) + biases
	l1 = tf.add(tf.mamul(data, hidden_layer_1['weights']), hidden_layer_1['biases']) 
	#activation function, Rectifier 
	l1 = tf.nn.relu(l1) 

	#layer 2, mnozi data iz hidden layer 1 sa weights od hidden_layer_2 te dodaje bias
	l2 = tf.add(tf.mamul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	#output layer, podaci sa layer 2 mnozi sa weights od output layer
	output_layer = tf.mamul(l2, output_layer['weights'], output_layer['biases'])

	return output_layer


#treniranje mreze
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#koliko puta ce ici back
	hm_epoch = 10
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in hm_epoch:
		




''' 
input > weight> hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

compare output to intended output > cost function (ex. cross entropy)
optimization function(optimizer) > minimize cost(AdamOptimizer, SGD, AdaGrad ect)

backpropagation

feedforward + backprop = epoch (koliko puta ce proci kroz mrezu)

'''