#encoding:Utf-8

'''요약'''
# NN형태
# L1=sig([784,256]X+[256])  -encoder1
# L2=sig([256,128]L1+[128]) -encoder2
# L3=sig([128,256]L2+[256]) -decoder1
# L4=sig([256,784]L3+[784]) -decoder2

# cost=reduce_mean(pow(Y_true-Y_pred,2))
# optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# NN에 임의의 값을 넣으면 같은 값을 출력하도록 하는 학습이다.
# L1,L2를 Encoder(Layer를 거칠수록 Output이 줄어듬)라고 하였고 L3,L4를 Decoder(Layer를 거칠수록 Output이 원상태로 복귀됨)라고 하였다.

'''질문'''
# 결국에 최종적으로 가장 큰 영향을 미치는 건 L2의 출력이지 않을까?
# y=WX+b의 단순한 형태로 학습을 시키는 것과 위의 방식으로 encoder와 decoder를 거쳐서 학습을 하는 것의 차이는 무엇일까?

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data", one_hot=True)

#Parameters
learning_rate=0.01
training_epochs=20
batch_size=256
display_step=1
example_to_show=10

#Network Parameters
n_hidden_1=256 #1st layer num feature
n_hidden_2=128 #2nd layer num feature
n_input=784 #MNIST data input(img shpae:28*28)

#tf graph input(only pictures)
X=tf.placeholder("float", [None, n_input])

#def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
 #                 seed=None, name=None):
"""Outputs random values from a normal distribution.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  """
#표준편차에 범위 내의 지정된 배열 형태로 랜덤 값을 출력해준다. 값을 미입력시 mean=0.0 편차는 1로 설정된다.

#weights와 biases를 Dictionary 형태로 저장 한다
weights={
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases={
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}

#y=WX+b 형태로 대입하는 코드
# W=tf.Variable(tf.random_normal([784,784]))

# b=tf.Variable(tf.random_normal([784]))
# my_pred=tf.nn.sigmoid(tf.add(tf.matmul(X,W),b))

#Building the encoder
def encoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']), biases['encoder_b2']))

    return layer_2

def decoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']), biases['decoder_b1']))

    #Decoder Hidden layer with sigmoid activation #2
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))

    return layer_2

#Construct model
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

#Prediction
y_pred=decoder_op
#Targets (Labels) are the input data.
y_true=X

#Define loss and optimizer, minimize the squared error
cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))

#y=WX+b 형태로 대입하는 코드
#cost=tf.reduce_mean(tf.pow(y_true-my_pred,2))

#RMSPropOptimizer를 사용한다(rmsprop: Divide the learning rate for a weight
# by a running average of the magnitudes of recent gradients for that weight.)
#magnitude 규모

optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
"""Optimizer that implements the RMSProp algorithm.

 See the [paper]
 (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
"""
#quadratic bowl 이차원 그릇 모양 parabola 포물선 ellipse 타원  approximation 근사치, 비슷한 것
#ravine 산골짜기, 협곡 oscillation 진동, 움직임  diverge 갈라지다, 나뉘다, 벗어나다
#simutaneously 동시에, 일제히 fluctuation 변동, 오르내림
#Initializing the variables
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    #Training cycle
    for epoch in range(training_epochs):
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            #Run optimization op (backprop) and cost op( to get loss value)
            _, c=sess.run([optimizer,cost], feed_dict={X:batch_xs})
        # Display logs per epoch step
        if epoch%display_step==0:
            print("Epoch", '%04d' %(epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

     #Applying encode and decode over test set
    encode_decode=sess.run(y_pred, feed_dict={X:mnist.test.images[:example_to_show]})
    # y=WX+b 형태로 대입하는 코드
    # encode_decode = sess.run(my_pred, feed_dict={X: mnist.test.images[:example_to_show]})
    #Compare original images with their reconstructions
    f, a=plt.subplots(2, 10, figsize=(10,2))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],[28,28]))


    f.show()
    plt.draw()
    plt.waitforbuttonpress()