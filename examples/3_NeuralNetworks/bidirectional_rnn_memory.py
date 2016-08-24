#encoding:Utf-8
'''
A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow libray
This example is using the MNIST database of handwritten digits
Long Short Term Memory paper:http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project : https://github.com/aymericdamien/TensorFlow-Examples/
'''



#bidirectional 양방향의 reccurent 반복되는
#bidirectional 양방향 LSTM방식이 적용된... 하지만 잘 모르겠다.

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
#기존의 tensorflow.model.rnn은 deprecated 되었다.
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

#Parameterws
learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

#Network Parameters
n_input=28 # MNIST data input (img shpae: 28*28)
n_steps=28 # timesteps
n_hidden=128 #hidden layer num of features
n_classes=10 #MNIST total classes(0-9 digits)

#tf Graph input
x=tf.placeholder("float", [None, n_steps, n_input])
y=tf.placeholder("float", [None, n_classes])

#Define weights
weights={
    #Hidden layer weights = 2*n_hidden because of forward + backward cells
    'out':tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases={
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x, weights, biases):

    #prepare data shape to match 'bidirectional_rnn' function requirements
    #Current data input shpae: (batch_size, n_steps, n_input)
    #Required shpae: 'n_steps' tensors list of shpae (batch_szie, n_input)

    #Permuting batch_size and n_steps

    #def transpose(a, perm=None, name="transpose"):
    """Transposes `a`. Permutes the dimensions according to `perm`.

    The returned tensor's dimension i will correspond to the input dimension
         `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
        the rank of the input tensor. Hence by default, this operation performs a
        regular matrix transpose on 2-D input Tensors.

        For example:

        ```python
        # 'x' is [[1 2 3]
        #         [4 5 6]]
        tf.transpose(x) ==> [[1 4]
                             [2 5]
                             [3 6]]

        # Equivalently
        tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                          [2 5]
                                          [3 6]]

        # 'perm' is more useful for n-dimensional tensors, for n > 2
        # 'x' is   [[[1  2  3]
        #            [4  5  6]]
        #           [[7  8  9]
        #            [10 11 12]]]
        # Take the transpose of the matrices in dimension-0
        tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                              [2  5]
                                              [3  6]]

                                             [[7 10]
                                              [8 11]
                                              [9 12]]]
        ```
    """
    #transpose 첫번째 parameter로 바꿀 행이고
    # 2번째 파라미터를 지정하지 않을 경우 n차원에 대하여(0, 1,2,3,..., n-2, n-1)를  (n-1, n-2, ... 2,1, 0)  형태로 변형
    # perm 을 지정시 (0, 1, 2) -> ([1, 0, 2])처럼 원하는 형태로 Rank를 바꿀수 있다. 예를들어
    # [[1,2][3,4]]  1/(0,0)  2/(0,1)   이것을  transpose 하면 (1, 0)순열로 transpse하면 (0,1)->(1,0)으로 바뀌므로
    #  1/(0,0) 2/(1,0)처럼 바뀐다 랭크를 1대 1 맵핑하여 이동시켜준다고 생각하면 편한다.
    #Permuting batch_size and n_steps
    x=tf.transpose(x,[1,0,2])

    #Reshape to (n_steps*batch_size, n_input)

    """Reshapes a tensor.

      Given `tensor`, this operation returns a tensor that has the same values
      as `tensor` with shape `shape`.

      If one component of `shape` is the special value -1, the size of that dimension
      is computed so that the total size remains constant.  In particular, a `shape`
      of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

      If `shape` is 1-D or higher, then the operation returns a tensor with shape
      `shape` filled with the values of `tensor`. In this case, the number of elements
      implied by `shape` must be the same as the number of elements in `tensor`.

      For example:

      ```prettyprint
      # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
      # tensor 't' has shape [9]
      reshape(t, [3, 3]) ==> [[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]

      # tensor 't' is [[[1, 1], [2, 2]],
      #                [[3, 3], [4, 4]]]
      # tensor 't' has shape [2, 2, 2]
      reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                              [3, 3, 4, 4]]

      # tensor 't' is [[[1, 1, 1],
      #                 [2, 2, 2]],
      #                [[3, 3, 3],
      #                 [4, 4, 4]],
      #                [[5, 5, 5],
      #                 [6, 6, 6]]]
      # tensor 't' has shape [3, 2, 3]
      # pass '[-1]' to flatten 't'
      reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

      # -1 can also be used to infer the shape

      # -1 is inferred to be 9:
      reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [4, 4, 4, 5, 5, 5, 6, 6, 6]]
      # -1 is inferred to be 2:
      reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [4, 4, 4, 5, 5, 5, 6, 6, 6]]
      # -1 is inferred to be 3:
      reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                    [2, 2, 2],
                                    [3, 3, 3]],
                                   [[4, 4, 4],
                                    [5, 5, 5],
                                    [6, 6, 6]]]

      # tensor 't' is [7]
      # shape `[]` reshapes to a scalar
      reshape(t, []) ==> 7
      ```
    """
    #reshape를 해준다 -1을 인자로 쓰면 나머치 차원을 알아서 추론해서 채워준다
    #[?(n_step*batch_size), 28] 형태로 reshape
    x=tf.reshape(x,[-1,n_input])

    #def split(split_dim, num_split, value, name="split"):
    """Splits a tensor into `num_split` tensors along one dimension.

        Splits `value` along dimension `split_dim` into `num_split` smaller tensors.
        Requires that `num_split` evenly divide `value.shape[split_dim]`.

        For example:

        ```python
        # 'value' is a tensor with shape [5, 30]
        # Split 'value' into 3 tensors along dimension 1
        split0, split1, split2 = tf.split(1, 3, value)
        tf.shape(split0) ==> [5, 10]
        ```

        Args:
          split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
            Must be in the range `[0, rank(value))`.
          num_split: A Python integer. The number of ways to split.
          value: The `Tensor` to split.
          name: A name for the operation (optional).

        Returns:
          `num_split` `Tensor` objects resulting from splitting `value`.
    """
    #Split to get a list of 'n_steps' tensor of shape (batch_size, n_input)
    #28개로 분할된 [None,28]로 된 28개의 x split로 나누어준다.
    x=tf.split(0,n_steps,x)
    #decay 썩다 부패하 multiplicative 증가하는 ㅊㅁ
    #Learning to store information over extended time intervals via recurrent backpropagation takes a very long time,
    #mostly, due to insufficient, decaying error back flow. We briefly review Hochreiter's 1991 analysis of this problem,
    #then address it by introducing a novel, efficient, gradient-based method called "Long Short-Term Memory"(LSTM). Truncating the gradient
    #where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000
    #special units. Multiplicative gate units learn to open and close accesss to the constant error
    #flow LSTM is local in space and time; its computational complexity per time step and weight is O(1). Our experiments with aritificial data
    #involve local, distributed, read-valued, and noisy pattern representations. In comparisons with RTRL, BPTT, Recurrent Cascade-Correlation,
    #Elman nets, and Neural Sequence Chunking, LSTM leads to many more sucessful runs, and learns much faster.
    #LSTM also solves complex, artificial long time lag takss that have never been solved by previous recurrent network algorithms


    lstm_fw_cell=rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell=rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    try:
          outputs, _, _=rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    except Exception:
          outputs=rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out'])+biases['out']

pred=BiRNN(x, weights, biases)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init=tf.initialize_all_variables()

with tf.Session() as sess:
      sess.run(init)
      step=1
      while step*batch_size< training_iters:
          batch_x, batch_y=mnist.train.next_batch(batch_size)
          batch_x=batch_x.reshape((batch_size, n_steps, n_input))
          sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
          if step%display_step==0:
              acc=sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
              loss=sess.run(cost, feed_dict={x:batch_x, y:batch_y})
              print "Iter"+str(step*batch_size)+", Minibatch Loss="+\
                  "{:.6f}".format(loss)+", Training Accuracy="+\
                  "{:.5f}".format(acc)
          step+=1
      print "optimization Finished"

      test_len=128
      test_data=mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
      test_label=mnist.test.labels[:test_len]
      print "Testing Accuracy:",\
          sess.run(accuracy, feed_dict={x:test_data, y:test_label})
