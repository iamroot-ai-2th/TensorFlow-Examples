#recurrent_netowrk
#BasicLSTM(n_hidden, forget)
#rnn(cell, x, ~)
#def Rnn(x, weights, bias)
#   2개의 층을 가지고 있음 LSTM을 해주는 층과 y=WX+b를 하는 층
#    1. LSTM(Projection layer) input(28*batch_size)의 steps list를 가지고 RNN을 생성 / n_hidden개의 결과를 출력
#    2. y=WX+B에 넣어줌 / softmax_cross_entropy_with_logits() / AdamOptimizer
# 하지만 여전히 LSTM에서 outputs의 리스트가 출력되는 과정이 이해가 가지 않습니다.
# LSTM Cell은 forget / +(x_current) / +(다음에 전달할 t 값 연산) : 이 3가지를 수행을 각 step별로 수행하여 다음에 영향을 미치고
# 결과를 출려하는 cell을 생성하게 되는데.. 여기에 어떻게 n_hidden 128이라는 값을 연결되어 있을까요?
# n_input 는 28*batch_size 그리고 step은 28개 이걸 가지고 ouputs[각 스텝의 결과들이 저장되어있는 리스트]를 반환
# 이걸 outputs[-1] 을 하면 2. y=WX+b 다음 연산에 사용할 수있는 X값이 된답니다. W=[n_hidden, 10] 이니까 X는  [n_hidden, 1]
# -> [10, 1]로 결과가 나옴
'''
A Reccurent Neural Netowrk(LSTM) implementation examle using Tensorflow libray.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author : Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a reccurent neural network, we consider every image row
as a sequence of pixels. Because MNIST image shape is 28*28px, we will then handle 28 sequnces of 28 steps for
every smaple.
'''

#Parameters
learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

#Network Parameters
n_input=28 #MNIST data input( img shape:28*28)
n_steps=28 #timesteps
n_hidden=128 #hidden layer num of features
n_classes=10 #MNIST total classes (0-9 digits)

#tf Graph input
x=tf.placeholder("float", [None, n_steps, n_input])
y=tf.placeholder("float", [None, n_classes])

# Define weights
weights={
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases={
    'out' : tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weigths, biases):

    #Prepare data shape to match 'rnn' function requirements
    #Current data input shape: (batch_size, n_steps, n_input)
    #Required shpae:'n_steps' tensors list of shape (batch_size, n_input)

    #Permuting batch size and n_steps
    x=tf.transpose(x,[1,0,2])
    #Reshaping to (n_steps*batch_size, n_input)
    x=tf.reshape(x, [-1, n_input])
    #Split to get a list of 'n_steps' tensors of shpae(batch_size, n_input)
    x=tf.split(0, n_steps, x)

    #Define a lstm cell with tensorflow
    lstm_cell=rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    #Get lstm cell output
    outputs, states=rnn.rnn(lstm_cell, x, dtype=tf.float32)

    #Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out'])+biases['out']

pred=RNN(x, weights, biases)

#Define loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model
correct_pred=tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initializing the Variables
init=tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step=1
    #Keep training until reach max iterations
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        #Reshape data to get 28 seq of 28 elements
        batch_x=batch_x.reshape((batch_size,n_steps, n_input))
        #Run optimizaiton op(bachProp)
        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
        if step%display_step==0:
            #Caculate batch accuracy
            acc=sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            #Caculate batch loss
            loss=sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            print "Iter"+str(step*batch_size)+", Minibatch Loss="+ \
                  "{:.6f}".format(loss)+", Training Accuracy ="+\
                  "{:.5f}".format(acc)
        step+=1
    print "Optimiation Finished!"

    #Caculate accuracy for 128 mnist test images
    test_len=128
    test_data=mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label=mnist.test.labels[:test_len]
    print "Testing Accuracy", \
        sess.run(accuracy, feed_dict={x:test_data, y:test_label})