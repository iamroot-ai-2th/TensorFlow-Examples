#coding:UTF-8
#한글을 쓰기위해서는 coding:UTF-8을 첫 라인에 작성해야 한다.

#tensorflow에서 가장 기초가 되는 개념은 tensor 객체의 개념이다. 모든 연산, 상수, 값에 대한 값을 저장할 때 tensor라는 객체로 저장되게 된다.
#가장 기초적으로 상수를 저장해서 이용 할 수 있다.


'''tensor의 개념과 기본적인 연산 처리'''
#먼저 tensorflow를 사용하기 위해서는 import를 하여야 한다. as 후에 나오는 이름을 별칭으로 사용하게 되서 tf라는 이름을 통해서 tensorflow 모듈을 이용할 수 있다.
import tensorflow as tf

a=tf.constant(4)
b=tf.constant(5)
c=tf.add(a,b)

#Tensor 객체로 선언되고 실질적인 run 함수의 호출 전까지 상수 연산 값들을 가지고 있게 된다.
print '\ntensor의 개념과 기본적인 연산 처리'
print 'Tensor 객체 a :', a, '\nTensor 객체 b: ', b,'\nTensor 객체 c :', c
print '\n'

#1. 세션은 무엇인가?
# A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are
#evaluated.
#하나의 세션 object는 연산이 수행되거나 Tensor objec가 평가되기 위한 환경을 모아놓은 객체이다.

#2. with tf.Session() as sess : 이것의 의미는?
#with tf.Session() as sess: 는
#sess=tf.Session()
#sess.run(...)
#sess.close()을 함축 하고 있다
# 들여쓰기가 끝나면 자동으로 세션의 종료를 호출해준다
with tf.Session() as sess:
    print 'sess.run(a) :', sess.run(a), '\nsess.run(b) :', sess.run(b), '\nsess.run(c) :', sess.run(c)
    print '\n'


'''placeholder와 feed_dict를 이용한 연산'''
X=tf.placeholder(tf.int16) #tf.int16크기의 빈공간을 만들고 X라는 이름을 붙인다.
Y=tf.placeholder(tf.int16)


add=tf.add(X,Y)
mul=tf.mul(X,Y)

with tf.Session() as sess:
    print 'placeholer와 feed_dict를 이용한 연산 결과'
    print 'sess.run(add, feed_dict={X:10, Y:10}) :', sess.run(add, feed_dict={X:10, Y:10})
    print 'sess.run(add, feed_dict={X:10, Y:10}) : ', sess.run(mul, feed_dict={X:10, Y:10})
    print '\n'


'''행렬을 이용한 연산'''
matrix1=tf.constant([[1.0, 1.0]])
matrix2=tf.constant([[1.0], [1.0]])

print '행렬을 이용한 연산'
print 'matrix1.get_shape() 함수 :', matrix1.get_shape()
print 'matrix1.get_shape().as_list() :', matrix1.get_shape().as_list()
print 'matrix2.get_shape().as_list() :',matrix2.get_shape().as_list()
product=tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print 'matrix1 * matrix2의 tensor 연산'
    print sess.run(matrix1), '\nX\n', sess.run(matrix2)
    print '=', sess.run(product)
    print '\n'


'''문자열의 값 저장과 출력'''
hello=tf.constant('Hello The World')

with tf.Session() as sess:
    print '문자열 저장과 출력'
    print sess.run(hello)
    print '\n'