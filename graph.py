#coding=utf-8
import numpy as np
import random
import copy
import pickle
import os
import chardet
class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Graph(object):
    def __init__(self, filepaths, num, input_size, path_length, times, id):
        self.V = num
        self.id = id
        self.adjacent_matrix = []
        self.global_adjacent_matrix = np.array([[0 for i in range(num)] for ii in range(num)])
        files = os.listdir(filepaths)
        filepaths = [filepaths+file for file in files]
        with open(filepaths[0],'r') as f:
            tmp = np.array([[0 for i in range(num)] for ii in range(num)])
            for line in f:
                line = line.strip()
                tmp[int(line.split('\t')[0])][int(line.split('\t')[1])] = 1
                tmp[int(line.split('\t')[1])][int(line.split('\t')[0])] = 1
                # self.global_adjacent_matrix[int(line.split(' ')[0])][int(line.split(' ')[1])] = 1
                # self.global_adjacent_matrix[int(line.split(' ')[1])][int(line.split(' ')[0])] = 1
            self.adjacent_matrix.append(tmp)
            # self.global_adjacent_matrix += self.adjacent_matrix
        for i in range(1, len(filepaths)):
            self.adjacent_matrix.append(copy.deepcopy(self.adjacent_matrix[-1]))
            with open(filepaths[i], 'r') as f:
                for line in f:
                    # print(chardet.detect(line))
                    line = line.strip()
                    tmp = line.split('\t')
                    change = 1
                    self.adjacent_matrix[-1][int(tmp[0])][int(tmp[1])] = change
                    self.adjacent_matrix[-1][int(tmp[1])][int(tmp[0])] = change
            # self.global_adjacent_matrix += self.adjacent_matrix[i] * (i + 1)
        for i in range(len(filepaths)):
            self.global_adjacent_matrix += self.adjacent_matrix[i] * (i+1)
        # self.adjacent_matrix = self.local_pattern(self.global_adjacent_matrix,
        #                                           path_length, times, num, input_size, len(filepaths))                 # path_length, times 填入



        # self.row_sum = np.sum(self.global_adjacent_matrix, axis=1)
        # self.probability = np.array([[0 for i in range(num)] for i in range(num)]) + self.global_adjacent_matrix
        # for i in range(num):
        #     for ii in range(num):
        #         self.probability[i][ii] /= self.row_sum[i]          #默认剔除孤立点

    def random_walk(self, G, path_length, start, times, num, rand=random.Random()):
        corpus = []
        for j in range(times):
            print("random walk "+str(j+1)+" times")
            res = []
            cur = start
            for i in range(path_length):
                # res.append(np.random.choice(num, 1, p=P[start])[0])
                try:
                    res.append(rand.choices([i for i in range(num)], k=1, weights=list(G[cur]))[0])
                except IndexError:
                    res.append(cur)
                cur = res[-1]
            corpus.append(res)
        d = {}
        for i in range(times):
            for node in corpus[i]:
                if node != start:
                    d[node] = d.setdefault(node, 0) + 1
        sorted_list = sorted(d.items(), key=lambda item: item[1], reverse=True)     # [(编号, 次数)]
        return [i[0] for i in sorted_list]

    def local_pattern(self, global_g, path_length, times, num, input_size, time_step):
        ind = []
        try:
            ff = open("randomWalk"+str(self.id)+".dat", 'rb')
            ind = pickle.load(ff)
        except:
            for i in range(num):
                print("node "+str(i)+" walking:")
                ind.append(self.random_walk(global_g, path_length, i, times, num))
            with open("randomWalk"+str(self.id)+".dat", "wb") as f:
                pickle.dump(ind, f)
        # local_adjacent_matrix = [np.array([[0 for i in range(input_size)] for i in range(num)])]
        # for i in range(1, time_step):
        #     local_adjacent_matrix.append(local_adjacent_matrix[-1])
        # local_adjacent_matrix = [np.array([[0 for i in range(input_size)] for ii in range(num)]) for iii in range(time_step)]
        local_adjacent_matrix = [[[0 for i in range(input_size)] for ii in range(num)] for iii in range(time_step)]

        for i in range(num):      #每个节点
            for ii in range(min(input_size, len(ind[i]))):    #每个节点的local list的每个element
                for iii in range(time_step):               #每个时间片
                    # print(iii, i, ii)
                    # print(len(self.adjacent_matrix), len(self.adjacent_matrix[0]), len(self.adjacent_matrix[0][0]))
                    # print(len(local_adjacent_matrix), len(local_adjacent_matrix[0]), len(local_adjacent_matrix[0][0]))
                    local_adjacent_matrix[iii][i][ii] = self.adjacent_matrix[iii][i][ind[i][ii]]
        for i in range(len(local_adjacent_matrix)):
            local_adjacent_matrix[i] = np.array(local_adjacent_matrix[i])
        return local_adjacent_matrix          # [time_step, |V|, lstm_input_size]

    def sample(self, st, batch_size, order, N):
        mini_batch = Dotdict()
        if st >= self.V:
            mini_batch.flag = True      # 采样超过本网络节点数
            index = order[0:1]
            mini_batch.global_adjacent_matrix = self.global_adjacent_matrix[index]
            mini_batch.adjacent_matrix = np.array([i[index] for i in self.adjacent_matrix])
            mini_batch.mini_batch_matrix = self.global_adjacent_matrix[index][:][:, index]
            return mini_batch
        V = N if st < N else self.V
        en = min(V, st + batch_size)
        index = order[st:en]
        mini_batch.global_adjacent_matrix = self.global_adjacent_matrix[index]
        mini_batch.adjacent_matrix = np.array([i[index] for i in self.adjacent_matrix])
        mini_batch.mini_batch_matrix = self.global_adjacent_matrix[index][:][:, index]
        mini_batch.flag = False
        return mini_batch





