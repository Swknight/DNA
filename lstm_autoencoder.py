import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Autoencoder():
    def __init__(self,
                 unit_num=6,
                 en_hidden_num=64,
                 de_hidden_num=64,
                 input_n_layer=1,
                 batch_size=5399,
                 embedding_length=64,
                 lr = 0.01
                 ):
        self._input_n_layer = input_n_layer
        self._unit_num = unit_num
        self._en_hidden_num = en_hidden_num
        self._de_hidden_num = de_hidden_num
        self._lr = lr
        self._batch_size = batch_size
        self._embedding_length = embedding_length



    def _build_graph(self):
        self.inputs_keep_prob=tf.placeholder(dtype=tf.float32)
        self.g_inputs = tf.placeholder(dtype=tf.float32,shape=(self._batch_size,self._unit_num,self._embedding_length),name = "en_inputs")
        self.d_inputs = tf.placeholder(dtype=tf.float32,shape=(self._batch_size,self._unit_num,self._embedding_length),name = "de_inputs")
        self.inputs = tf.unstack(self.g_inputs,axis=1)
        self.deinputs = tf.unstack(self.d_inputs,axis=1)
        self.V = tf.placeholder(dtype=tf.float32,shape=(None,None))
        self.Q = tf.placeholder(dtype=tf.float32,shape=(None,None))
        self._adj_matrix = tf.placeholder(dtype=tf.float32,shape = (None,None))
        def _get_lstm_cells(hidden_layer):
            return tf.nn.rnn_cell.LSTMCell(num_units=hidden_layer,activation=tf.nn.relu)

        with tf.variable_scope("encoder") as encoder:

            cells_list = []
            for num in range(self._input_n_layer):
                cells_list.append(_get_lstm_cells(hidden_layer= self._en_hidden_num))
            for ind_input_lstm,each_input_lstm in enumerate(cells_list):
                cells_list[ind_input_lstm] = tf.contrib.rnn.DropoutWrapper(each_input_lstm,output_keep_prob = self.inputs_keep_prob)
            cells= tf.contrib.rnn.MultiRNNCell(cells_list)
            (enc_outputs,enc_state) = tf.contrib.rnn.static_rnn(cells,self.inputs,dtype=tf.float32)

        self.H = enc_outputs[-1]

        with tf.variable_scope("decoder") as decoder:

            de_cells_list = []
            for num in range(self._input_n_layer):
                de_cells_list.append(_get_lstm_cells(hidden_layer=self._de_hidden_num))
            for ind_input_lstm, each_input_lstm in enumerate(de_cells_list):
                de_cells_list[ind_input_lstm] = tf.contrib.rnn.DropoutWrapper(each_input_lstm,
                                                                           output_keep_prob=self.inputs_keep_prob)
            de_cells = tf.contrib.rnn.MultiRNNCell(de_cells_list)
            # self._initial_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(self.H,self.H)for idx in range(self._input_n_layer)])
            (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(de_cells, self.deinputs,initial_state=enc_state,dtype=tf.float32)
            self.outputs = dec_outputs[::-1]

        D = tf.diag(tf.reduce_sum(self._adj_matrix,1))
        L = D - self._adj_matrix
        self.loss1 = tf.norm((tf.subtract(self.inputs,self.outputs)))
        self.loss2 = tf.norm((tf.subtract(self.inputs,self.outputs)))+tf.norm(tf.subtract(self.H,tf.matmul(self.V,self.Q)))
        # 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.H), L), self.H))
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #loss1
        optimizer1 = tf.train.RMSPropOptimizer(learning_rate=self._lr)
        raw_gard1 = tf.gradients(self.loss1,train_vars)
        clip_gard1,_ = tf.clip_by_global_norm(raw_gard1,5)
        self.rms_train_op1 = optimizer1.apply_gradients(zip(clip_gard1,train_vars))

        #loss2
        optimizer2 = tf.train.RMSPropOptimizer(learning_rate=self._lr)
        raw_gard2 = tf.gradients(self.loss2, train_vars)
        clip_gard2, _ = tf.clip_by_global_norm(raw_gard2, 5)
        self.rms_train_op2 = optimizer2.apply_gradients(zip(clip_gard2, train_vars))

        self.init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(self.init)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./tensorflow_autoencoder_1.0_logs', sess.graph)
    # def get_data(self,filepath,num,id):
    #     g = Graph(filepath,num,64,50,10,1)
    #     f_adj_matrix = g.global_adjacent_matrix
    #     f_data = np.zeros(shape=(num,6,64))
    #     for i in range(6):
    #         f = sio.loadmat('mat/graph' + str(id) + 'piece' + str(i + 1) + '.mat')
    #         f_d = f['matrix']
    #         f_row_max = np.max(f_d, axis=1)
    #         f_row_max[f_row_max == 0] = 1
    #         trans_f_row_max = np.transpose([f_row_max])
    #         f_d = f_d / trans_f_row_max
    #         f_data[:, i, :] = f_d
    #     return f_adj_matrix,f_data


    # def train(self,auto=20,epoch=10):
    #     filepath = []
    #     filepath.append("/Users/jianwen/Documents/python/Autoencoder/f-piece/")
    #     filepath.append("/Users/jianwen/Documents/python/Autoencoder/t-piece/")
    #     nums = [5399, 5326]
    #     for i in range(2):
    #         if i == 0:
    #             f_adj_matrix, f_data = self.get_data(filepath[i], nums[i], i+1)
    #         if i == 1:
    #             t_adj_matrix, t_data = self.get_data(filepath[i], nums[i], i+1)
    #     with tf.variable_scope("network1") as n1:
    #         model1 = Autoencoder()
    #         model1._build_graph()
    #     with tf.variable_scope("network2") as n2:
    #         model2 = Autoencoder()
    #         model2._build_graph()
    #     with tf.Session() as sess:
    #         for one_auto in range(auto):
    #             if one_auto == 0:
    #                 for one_epoch in range(epoch):
    #                     print("---------------------------EPOCH" + str(one_epoch) + "-----------------------")
    #                     rms_train,lossValue,model1_U = sess.run([model1.rms_train_op1,model1.loss1,model1.H],feed_dict={model1.g_inputs:f_data,model1.inputs_keep_prob:0.8,model1._adj_matrix:f_adj_matrix})
    #                     print("---------------------------Loss" + str(lossValue) + "-----------------------")
    #                 for one_epoch in range(epoch):
    #                     print("---------------------------EPOCH" + str(one_epoch) + "-----------------------")
    #                     rms_train,lossValue,model2_U = sess.run([model2.rms_train_op1,model2.loss1,model2.H],feed_dict={model2.g_inputs:f_data,model2.inputs_keep_prob:0.8,model2._adj_matrix:f_adj_matrix})
    #                     print("---------------------------Loss" + str(lossValue) + "-----------------------")
    #                 nto1_1 = Nto1(model1_U,model2_U,128,nums[0])
    #                 v1 = nto1_1.v_op()
    #                 v1 = v1.eval()
    #                 q1 = nto1_1.q_op()
    #                 q1 = q1.eval()
    #                 nto1_2 = Nto1(model2_U, model1_U, 128, nums[0])
    #                 v2 = nto1_2.v_op()
    #                 v2 = v2.eval()
    #                 q2 = nto1_2.q_op()
    #                 q2 = q2.eval()
    #             else:
    #                 for one_epoch in range(epoch):
    #                     print("---------------------------EPOCH" + str(one_epoch) + "-----------------------")
    #                     rms_train, lossValue,model1_U = sess.run([model1.rms_train_op2, model1.loss2,model1.H],feed_dict={model1.g_inputs: f_data, model1.inputs_keep_prob: 0.8,model1._adj_matrix: f_adj_matrix,model1.V:v1,model1.Q:q1})
    #                     print("---------------------------Loss" + str(lossValue) + "-----------------------")
    #                 for one_epoch in range(epoch):
    #                     print("---------------------------EPOCH" + str(one_epoch) + "-----------------------")
    #                     rms_train, lossValue,model2_U = sess.run([model2.rms_train_op2, model2.loss2,model2.H],feed_dict={model2.g_inputs: f_data, model2.inputs_keep_prob: 0.8,model2._adj_matrix: f_adj_matrix,model2.V:v2,model2.Q:q2})
    #                     print("---------------------------Loss" + str(lossValue) + "-----------------------")
    #                 nto1_1.U1 = nto1_1.set_U(model1_U)
    #                 v1 = nto1_1.v_op()
    #                 v1 = v1.eval()
    #                 q1 = nto1_1.q_op()
    #                 q1 = q1.eval()
    #                 nto1_2.U1 = nto1_2.set_U(model2_U)
    #                 v2 = nto1_2.v_op()
    #                 v2 = v2.eval()
    #                 q2 = nto1_2.q_op()
    #                 q2 = q2.eval()










    #
    # def train(self,auto=20,epoch=10):
    #     filepath = []
    #     filepath.append("/Users/jianwen/Documents/python/Autoencoder/f-piece/")
    #     filepath.append("/Users/jianwen/Documents/python/Autoencoder/t-piece/")
    #     nums = [5399,5326]
    #     for i in range(2):
    #         if i ==0:
    #             f_adj_matrix,f_data = self.get_data(filepath[i],nums[i],i+1)
    #     with tf.Session() as sess:
    #         sess.run(self.init)
    #
    #         for one_auto in range(auto):
    #             print("---------------------------AUTO" + str(one_auto) + "-----------------------")
    #             if one_auto == 0:
    #                 for one_epoch in range(epoch):
    #                     print("---------------------------EPOCH" + str(one_epoch) + "-----------------------")
    #                     rms_train,lossValue,U = sess.run([self.rms_train_op1,self.loss1,self.H],feed_dict={self.g_inputs:f_data,self.inputs_keep_prob:0.8,self._adj_matrix:f_adj_matrix})
    #                     print("---------------------------Loss" + str(lossValue) + "-----------------------")
    #                 nto1 = Nto1(U,128,5399)
    #                 v = nto1.v_op()
    #                 v = v.eval()
    #                 q = nto1.q_op()
    #                 q = q.eval()
    #             else:
    #                 for one_epoch in range(epoch):
    #                     print("---------------------------EPOCH" + str(one_epoch) + "-----------------------")
    #                     rms_train, lossValue,U = sess.run([self.rms_train_op2, self.loss2,self.H],feed_dict={self.g_inputs: f_data, self.inputs_keep_prob: 0.8,self._adj_matrix: f_adj_matrix,self.V:v,self.Q:q})
    #                     print("---------------------------Loss" + str(lossValue) + "-----------------------")
    #                 nto1.set_U(U)
    #                 v = nto1.v_op()
    #                 v = v.eval()
    #                 q = nto1.q_op()
    #                 q = q.eval()
    #             np.savetxt("embedding1.txt",U)
    #             tf.summary.scalar("loss",lossValue)
    #         # saver = tf.train.Saver()
    #         # saver.save(sess,"autoencoder_model")
    #         # print ("Save the model!")
    #         merged = tf.summary.merge_all()
    #         writer = tf.summary.FileWriter('./tensorflow_autoencoder_1.0_logs', sess.graph)



# if __name__ == '__main__':
#     model = Autoencoder()
#     print("-------------------------Build Graph---------------------------------")
#     model._build_graph()
#     print("-------------------------Start Training------------------------------")
#     model.train(epoch=100,auto=50)