#encoding:Utf-8
#Perceptron 학습 능력을 갖는 패턴 분류장치(입력층, 연합층 출력층을 가지며 신경세포를 모델화하였다.)

# multi layer peerceptron
# n_input 784 / n_hidden_1 256 / n_hidden_2 256 / n_classes 10
# Relu  Activation Funciton

# def multilayer_perceptron(x, weights, biases) / dictionary for W, B
# softmax_cross_entropy_with_logits(pred, y)
# tf.train.AdamOptimizer


'''
A Multilyaer Perceptron implementation example using Tensorflow libray.
This examples is using the MNIST database of handwritten digits
'''

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

#Parameters
learning_rate=0.001
training_epochs=30
batch_size=100
display_step=1

#Netowrk Parameters
n_hidden_1=256 #1st layer number of features
n_hidden_2=256 #2nd layer number of features
n_input=784 # MNSIT data input
n_classes=10 #MNIST total classes(0-9digits)

#tf Graph input
x=tf.placeholder("float", [None, n_input])
y=tf.placeholder("float", [None, n_classes])

#Create model
def multilayer_perceptron(x, weights, biases):
    #Hidden layer iwth RELU activation
    layer_1=tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    #Hidden layer with RELU activation
    layer_2=tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    #Output layer with linear activation
    out_layer=tf.matmul(layer_2, weights['out'])+biases['out']
    return out_layer

#Store layers weight & bias
weights={
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

#Construct model
pred=multilayer_perceptron(x,weights, biases)

# Define loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
"""softmax_cross_entropy_with_logits(pred,y)
Computes softmax cross entropy between `logits` and `labels`.

 Measures the probability error in discrete classification tasks in which the
 classes are mutually exclusive (each entry is in exactly one class).  For
 example, each CIFAR-10 image is labeled with one and only one label: an image
 can be a dog or a truck, but not both."""

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#([pdf](http://arxiv.org/pdf/1412.6980.pdf)).  -> AdamOptimizer

#Initialzing the variables
init=tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)

    #Training cycle
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y=mnist.train.next_batch(batch_size)

            #Run optimization op (backprop) and cost op(to get loss value)
            _, c=sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})

            #Compute average loss
            avg_cost+=c/total_batch

        #Display logs per epoch step
        if epoch %display_step==0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimizaiton Finished!"

    #Test model
    correct_prediction=tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

    #Caculate accuracy
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

    print "Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})