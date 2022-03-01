# 根据参数配置节点
from matplotlib import pyplot as plt
import numpy as np
import random
import math
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)

same_seeds(2048)

class system_model():
    def __init__(self, node_num, node_e):
        self.node_num = node_num
        self.node_e = node_e

    def init_nodes(self):
        node_pos_matrix = list(range(self.node_num*self.node_num))

        res = random.sample(node_pos_matrix, self.node_num)

        for i in range(self.node_num*self.node_num):
            if i in res:
                node_pos_matrix[i] = 1
            else:
                node_pos_matrix[i] = 0

        node_pos_matrix = np.array(node_pos_matrix).reshape(self.node_num, self.node_num)
        # print("node_position:\n", node_pos_matrix)
        positions = np.where(node_pos_matrix == 1)
        positions = [[x, y] for x, y in zip(positions[0], positions[1])]
        # print(positions)

        dis_matrix = []
        for i, pos_1 in enumerate(positions):
            tmp = []
            for j, pos_2 in enumerate(positions):
                dist_tp = math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)
                tmp.append(dist_tp)
            dis_matrix.append(tmp)
        dis_matrix = np.array(dis_matrix)

        dist_record = []
        for i in range(self.node_num):
            tmp_dist = np.sum(dis_matrix[i,range(self.node_num)])
            dist_record.append(tmp_dist)
        central_node_index = np.argmin(dist_record)
        central_sink = positions[central_node_index]

        positions.pop(central_node_index)
        sinks = [[np.array(pos)]+[self.node_e] for pos in positions]
        self.nodes = {
            "central_sink": np.array(central_sink),
            "sinks": sinks
        }
        # print(self.nodes)

    def show_nodes(self):
        plt.figure()
        central_pos = self.nodes["central_sink"]
        plt.scatter([central_pos[0]], [central_pos[1]],marker="*", s=100, c="r", alpha=1)

        sinks_pos_x = [sink[0][0] for sink in self.nodes["sinks"]]
        sinks_pos_y = [sink[0][1] for sink in self.nodes["sinks"]]
        sinks_s = [30*sink[1]/self.node_e for sink in self.nodes["sinks"]]
        plt.scatter(sinks_pos_x,sinks_pos_y,s=sinks_s,c="g",alpha=0.5,label="sink")

        plt.show()


if __name__ == '__main__':
    s_model = system_model(20, 30)
    s_model.init_nodes()
    s_model.show_nodes()
