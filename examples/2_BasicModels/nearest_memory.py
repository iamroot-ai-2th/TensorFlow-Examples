import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

Xtr, Ytr=mnist.train.next_batch(5000)
Xte, Yte=mnist.test.next_batch(200)

xtr=tf.placeholder("float", [None, 784])
xte=tf.placeholder("float", [784])
#5000개의 데이터 각각에 대해서 각필셀에대해 |xtr-xte|후  reduce_sum을 하여  (5000,)의 ndarray로 저장을한다
#각가의 픽셀에 대해서 다른 값을 가지고 있을 수록 값이 커지게 된다.
distance=tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))),reduction_indices=1)

#0차원의 값 중에 가장 낮은 값을 가진 값의 index를 반환한다
pred=tf.arg_min(distance, 0)

accuracy=0.

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        #nn_index의 i번째 테스트용 데이터와 가장 비슷한 값을 가진  훈련용데이터의 인덱스가 반환된다.
        nn_index=sess.run(pred, feed_dict={xtr:Xtr, xte:Xte[i]})

        print "Test", i, "Prediction:", np.argmax(Ytr[nn_index]),\
        "True Class : ", np.argmax(Yte[i])

        if np.argmax(Ytr[nn_index])==np.argmax(Yte[i]):
            accuracy+=1./len(Xte)

    print "Done!"
    print "Accuracy:", accuracy

