#coding=utf-8
import tensorflow as tf
import numpy as np
import  random

class Nto1(object):
    def __init__(self,U1,U2,hidden_num,num1,num2):
        self._hidden_num = hidden_num
        self._num1 = num1
        self._num2 = num2
        self.U1 = self.set_U(U1)
        self.U2 = self.set_U(U2)
        self.V1 = self.U1
        self.V2 = self.U2
        self.Q = tf.eye(self._hidden_num,dtype=np.float32)


    def v_op(self):
        pu = tf.matmul(self.U1,tf.transpose(self.Q))
        self.psi = (tf.add(tf.abs(pu),pu))/2
        self.upsilon = tf.subtract(tf.abs(pu),pu)/2
        pg = tf.matmul(self.Q,tf.transpose(self.Q))
        self.phi = (tf.abs(pg)+pg)/2
        self.gamma = (tf.abs(pg)-pg)/2
        idx = list(range(3380))
        random.shuffle(idx)
        label = tf.expand_dims(tf.constant(idx[:int(3380*0.8)]),1)
        index = tf.expand_dims(tf.range(0,int(3380*0.8)),1)
        concated = tf.concat((index,label),1)
        self.p1 = tf.sparse_to_dense(concated,[int(3380*0.8),self._num1],1.0,0.0)
        self.p2 = tf.sparse_to_dense(concated,[int(3380*0.8),self._num2],1.0,0.0)
        self.pi = tf.matmul(tf.matmul(tf.transpose(self.p1),self.p1),self.V1)
        self.lamb = tf.matmul(tf.matmul(tf.transpose(self.p1),self.p2),self.V2)
        self.V11 = tf.sqrt(tf.divide(tf.add(tf.add(self.psi,tf.matmul(self.V1,self.gamma)),self.lamb) , tf.add(tf.add(self.upsilon,tf.matmul(self.V1,self.phi)),self.pi)))
        self.V1 = tf.multiply(self.V1,self.V11)
        return self.V1

    def q_op(self):
        # print(tf.matmul(tf.transpose(self.V),self.V))
        # print(self.V)
        # print("Q")
        # print(self.Q)
        # print("U")
        # print(self.U)
        # with tf.Session() as sess:
            # print("UUUUUUUUUU")
            # print(sess.run(self.U))
            # print(sess.run(self.V1))
            # print ("VVVVVVVVVVV")
            # print(sess.run(self.V))
            # print(sess.run(tf.matmul(tf.transpose(self.V),self.V)))
        self.Q = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.V1),self.V1)),tf.transpose(self.V1)),self.U1)
        return self.Q

    def set_U(self,new_U):
        new_U[new_U == 0] = np.min(new_U[new_U != 0])/10   #zero to minimum value
        U = tf.convert_to_tensor(new_U)
        return U

if __name__ == "__main__":
    a = Nto1([1,2,3],3)
    a.v_op()