#coding:UTF-8

'''linear_regression 선형 회귀 문제 Y=ax+b 그래프를 갖는 문제에 대한 학습'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt


rng=numpy.random

'''parameters'''
learning_rate=0.01
training_epochs=1000
display_step=50


'''훈련용 데이터'''

train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#train_X.shape[0] 배열값의 갯수를 알아 낼 때 사용 할 수 있다.
n_samples=train_X.shape[0] #0차원 값의 갯 수


# numpy.asarray함수 Convert the input to an array(input을 ndarray(N-Dimensional array) type으로 바꾼다)
# a=[1,2]
# numpy.asarray(a)

list_x= [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1]

np_ndarray=numpy.asarray([[3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1]])
print type(list_x)
print type(np_ndarray)

#Y=WX+b를 위한 placeholder와 Variables를 설정한다.
X=tf.placeholder('float')
Y=tf.placeholder('float')

W=tf.Variable(1.0, name='weight')
b=tf.Variable(1.0, name='bias')

#Y=WX+b형태의 학습 모델을 만든다.
pred=tf.add(tf.mul(X,W),b)

#(WX+b-Y)제곱 /2*n_samples (Mean squared error)
#이것을 W에 관한 식으로 변환하면 W에 대한 2차함수의 식이 나오게 되서 U자 양의 그래프를 그리게 된다
# 그래서 U자의 가장 밑부분의 값을 가지게 되는 W 값이 최적의 결과값이 되게 된다.
cost=tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#tf.initialize_all_variables()함수를 통해서 tensor의 모든 값들을 초기화한다
init=tf.initialize_all_variables()

#cost의 경우 reduce_sum 함수를 통해서 모든 객체들에 대한 reducing 연산을 사용하고 있다
#그래서 sess.run(cost, feed_dict={X:train_X, Y:train_Y})를 통해서 연산을 하게 된다
#optimizer의 경우 위에서 reduce연산을 하기 때문에 여러개의 값을 넣을 수 있고 하나의 값을 넣을 수도 있다.
#하나씩 넣는 것보다 연산 속도면에서 여러개를 넣는 것이 더 좋았다.
#standar gradient descent(전체 train set으로 학습하는것) vs stochastic gradient descent (1개 또는 소수 단위로 학습하는 것)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X:train_X, Y:train_Y})
        # if (epoch+1)%display_step==0:
        #     c=sess.run(cost, feed_dict={X:train_X, Y:train_Y})
        #     print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
        #         "W=", sess.run(W), "b=", sess.run(b)
    if (epoch+1)%display_step==0:
        c=sess.run(cost, feed_dict={X:train_X, Y:train_Y})
        print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
              "W=", sess.run(W), "b=", sess.run(b)
    # pyplot에 관한 참고사이트
    # http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
    #Graph로 표현하기
    # plt.plot(train_X, train_Y, 'ro', label='Original data')
    # plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label='Fitted Line')
    # plt.legend()
    # plt.savefig('train.png')

#
# test_X = numpy.asarray([1.1, 3.1, 4.1])
# test_Y = numpy.asarray([2.1, 3.1, 4.1])
#
# plt.plot(test_X, test_Y, 'ro', label='Original data')
# plt.plot(test_X, 1*test_X, label='line')
# plt.legend()
# plt.savefig('train2.png')

# x=numpy.arange(0, 5, 0.1)
# y=numpy.sin(x)
# print type(x),type(y)
# plt.plot(x,y)
# plt.plot(x,y, 'ro')
# plt.legend()
# plt.savefig('train2.png')

    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    testing_cost=sess.run(tf.reduce_sum(tf.pow(pred-Y,2))/(2*test_X.shape[0]), feed_dict={X:test_X, Y:test_Y})
    print "test cost=", "{:.9f}".format(testing_cost), \
        "W=", sess.run(W), "b=", sess.run(b)

    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label='Fitted Line')
    # plt.legend()
    # plt.savefig('test2.png')