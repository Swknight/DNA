from lstm_autoencoder import Autoencoder
from graph import Graph
from Nto1 import Nto1
import scipy.io as sio
import numpy as np
import tensorflow as tf

class Main():

    def get_data(self,filepath,num,id):
        g = Graph(filepath,num,64,50,10,1)
        f_adj_matrix = g.global_adjacent_matrix
        f_data = np.zeros(shape=(num,6,64))
        for i in range(6):
            f = sio.loadmat('mat/graph' + str(id) + 'piece' + str(i + 1) + '.mat')
            f_d = f['matrix']
            f_row_max = np.max(f_d, axis=1)
            f_row_max[f_row_max == 0] = 1
            trans_f_row_max = np.transpose([f_row_max])
            f_d = f_d / trans_f_row_max
            f_data[:, i, :] = f_d
        return f_adj_matrix,f_data


    def train(self, auto=20, epoch=10):
        filepath = []
        filepath.append("/Users/jianwen/Documents/python/Autoencoder/f-piece/")
        filepath.append("/Users/jianwen/Documents/python/Autoencoder/t-piece/")
        nums = [5399,5326]
        for i in range(2):
            if i == 0:
                f_adj_matrix, f_data = self.get_data(filepath[i], nums[i], i + 1)
            if i == 1:
                t_adj_matrix, t_data = self.get_data(filepath[i], nums[i], i + 1)
        de_f_data = np.zeros((nums[0], 6, 64))
        de_f_data[:, 1:6, :] = f_data[:, ::-1, :][:, 0:5, :]

        de_t_data = np.zeros((nums[1], 6, 64))
        de_t_data[:, 1:6, :] = t_data[:, ::-1, :][:, 0:5, :]

        print("-------------------------Build Graph---------------------------------")
        with tf.variable_scope("network1") as n1:
            model1 = Autoencoder(batch_size=nums[0])
            model1._build_graph()
        with tf.variable_scope("network2") as n2:
            model2 = Autoencoder(batch_size=nums[1])
            model2._build_graph()
        print("-------------------------Start Training------------------------------")
        with tf.Session() as sess:
            sess.run(model1.init)
            sess.run(model2.init)
            for one_auto in range(auto):
                if one_auto == 0:
                    for one_epoch in range(epoch):
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk1----------EPOCH：" + str(one_epoch) + "-----------------------")
                        rms_train1, lossValue1, model1_U = sess.run([model1.rms_train_op1, model1.loss1, model1.H],
                                                                  feed_dict={model1.g_inputs: f_data,
                                                                             model1.d_inputs:de_f_data,
                                                                             model1.inputs_keep_prob: 0.8,
                                                                             model1._adj_matrix: f_adj_matrix})
                        print(model1_U)
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk1----------Loss：" + str(lossValue1) + "-----------------------")
                    for one_epoch in range(epoch):
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk2-----------EPOCH：" + str(one_epoch) + "-----------------------")
                        rms_train2, lossValue2, model2_U = sess.run([model2.rms_train_op1, model2.loss1, model2.H],
                                                                  feed_dict={model2.g_inputs: t_data,
                                                                             model2.d_inputs:de_t_data,
                                                                             model2.inputs_keep_prob: 0.8,
                                                                             model2._adj_matrix: t_adj_matrix})
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk2----------Loss：" + str(lossValue2) + "-----------------------")
                    nto1_1 = Nto1(model1_U, model2_U, 64, nums[0],nums[1])
                    v1 = nto1_1.v_op()
                    v1 = v1.eval()
                    q1 = nto1_1.q_op()
                    q1 = q1.eval()
                    nto1_2 = Nto1(model2_U, model1_U, 64, nums[1],nums[0])
                    v2 = nto1_2.v_op()
                    v2 = v2.eval()
                    q2 = nto1_2.q_op()
                    q2 = q2.eval()
                else:
                    for one_epoch in range(epoch):
                        print("-------------Auto:"+str(one_auto)+"---------Netowrk1----------EPOCH：" + str(one_epoch) + "-----------------------")
                        rms_train1, lossValue1, model1_U = sess.run([model1.rms_train_op2, model1.loss2, model1.H],
                                                                  feed_dict={model1.g_inputs: f_data,
                                                                             model1.d_inputs:de_f_data,
                                                                             model1.inputs_keep_prob: 0.8,
                                                                             model1._adj_matrix: f_adj_matrix,
                                                                             model1.V: v1, model1.Q: q1})
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk1----------Loss：" + str(lossValue1) + "-----------------------")
                    for one_epoch in range(epoch):
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk2----------EPOCH：" + str(one_epoch) + "-----------------------")
                        rms_train2, lossValue2, model2_U = sess.run([model2.rms_train_op2, model2.loss2, model2.H],
                                                                  feed_dict={model2.g_inputs: t_data,
                                                                             model2.d_inputs:de_t_data,
                                                                             model2.inputs_keep_prob: 0.8,
                                                                             model2._adj_matrix: t_adj_matrix,
                                                                             model2.V: v2, model2.Q: q2})
                        print("--------------Auto:"+str(one_auto)+"---------Netowrk2----------Loss：" + str(lossValue2) + "-----------------------")
                    nto1_1.U1 = nto1_1.set_U(model1_U)
                    v1 = nto1_1.v_op()
                    v1 = v1.eval()
                    q1 = nto1_1.q_op()
                    q1 = q1.eval()
                    nto1_2.U1 = nto1_2.set_U(model2_U)
                    v2 = nto1_2.v_op()
                    v2 = v2.eval()
                    q2 = nto1_2.q_op()
                    q2 = q2.eval()
            np.savetxt("embedding1.txt",model1_U)
            np.savetxt("embedding2.txt",model2_U)
if __name__ == '__main__':
    model_main = Main()
    model_main.train(epoch=100,auto=5)
