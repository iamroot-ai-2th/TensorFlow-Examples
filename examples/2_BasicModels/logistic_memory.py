#encoding:UTF-8

#논리 회귀 문제 0 or 1의 결과를 가지는 회귀 문제에 대한 솔루션이다
# 0~9의 값의 그림에 대하여 그 값이 틀리면 0 맞으면 1인 one hot vector에 대한 결과  layer를 이용하여
# y=wx+b형태의 메트릭스 곱에 대해 w와  b값을 학습 시켜서  y를 예측하는 프로그램을 만든다.

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


#tensorflow image와 label데이터를 Datasets 형태로 저장한다
#mnist에는 train, validation, test에 각각의 image와 label에 대한 데이터를 저장하고 있다.
# print mnist.test.images.shape[0]
# print mnist.test.labels.shape[0]
#
# print mnist.validation.images.shape[0]
# print mnist.validation.labels.shape[0]
#
# print mnist.train.images.shape[0]
# print mnist.train.labels.shape[0]

mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

'''n*n 형태로 여러개의 이미지를 보는 코드'''
# fig, ax=plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax=ax.flatten()
#
# for i in range(10):
#     img=mnist.test.images[i]. reshape([28,28])
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     plt.tight_layout()
#     plt.savefig("mnist.png")
# exit(0)

'''이미지 하나를 보는 코드'''
# img=mnist.test.images[0]. reshape([28,28])
# plt.imshow(img, cmap='Greys', interpolation='nearest')
# plt.show()
# exit(0)

learning_rate=0.01
training_epochs=25
batch_size=100
display_step=1

x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32, [None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))


''' softmax함수의 동작 방식
For each batch `i` and class `j` we have

      softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))'''
pred=tf.nn.softmax(tf.matmul(x,W)+b)


'''Computes the mean of elements across dimensions of a tensor.

    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
'''
# y의 값이 0일 경우에는 cost에 영향을 미치지 않는다 1일 경우만 log(pred)한 값이 cost의 값에 영향을 미치게 된다.
# reduction_indices가 1인 이유는 0차원의 값이 None이기 때문이다.
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch) :
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            _, c=sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})

            avg_cost+=c/total_batch

        if (epoch+1)% display_step==0:
            print "Epoch:", '%04d' %(epoch+1), "cost=", "{:.9f}".format(avg_cost)

    #argmax : Returns the index with the largest value across dimensions of a tensor.
    #input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    #dimension: A `Tensor` of type `int32`.
    #  int32, 0 <= dimension < rank(input).  Describes which dimension
    # of the input Tensor to reduce across. For vectors, use dimension = 0.
    #각 tensor의 1차원의 값중 가장 큰 값의 index를 반환하고 같으면 True를  다르면 False를 반환한다
    correct_prediction=tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))

    #Casts a tensor to a new type.
    #True, False bool type을 tf.float32형태로 형변화 시키고 그것을 reduce_mean연산을 한다. 1,0,1,0 -> 2/4 -> 0.5
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #eval
    #Calling this method will execute all preceding operations that
    #produce the inputs needed for the operation that produces this
    #tensor.

    #N.B.* Before invoking `Tensor.eval()`, its graph must have been
    #launched in a session, and either a default session must be
    #available, or `session` must be specified explicitly.
    #eval이라는 함수를 sess.run을 호출한다. 인자로 feed_dict와 sess를 가질 수 있다.
    print "Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
