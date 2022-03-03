import numpy as np
from matplotlib import pyplot as plt
import math
from settings import *
colors = ["r", "g", "b", "y", "m", "c", "w", "k"]


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001,max_iter=1000, only_dis=False):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.only_consider_dis = only_dis


    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i][0]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    if self.only_consider_dis:
                        distances.append(np.linalg.norm(feature[0] - self.centers_[center]))
                    else:
                        distances.append((np.linalg.norm(feature[0] - self.centers_[center]))/feature[1]) # e/dis
                classification = distances.index(min(distances))

                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)

            for c in self.clf_:
                if self.clf_[c] == []:
                    self.centers_[c] = np.average(np.array(self.clf_[c]), axis=0)
                else:
                    self.centers_[c] = np.average(np.array(self.clf_[c])[:,0], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False

            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index

def custom_k_means(x,max_k = 6,verbose=False):
    stds = []
    centers = []
    cls = []
    min_k = 2 if len(x) > 2 else 1
    for k in range(min_k, max_k):
        k_means = K_Means(k=k)
        k_means.fit(x)
        cls_e_dis = []
        for c in k_means.clf_:
            cls_e = 0
            for feature in k_means.clf_[c]:
                for i, pos in enumerate(x):
                    if (pos[0] == feature[0]).all():
                        cls_e += feature[1]
                        break
            cls_e_dis.append(cls_e)

        if verbose:
            # 绘制聚类过程的图
            plt.figure()
            plt.title(f"k={k}")
            for center in k_means.centers_:
                plt.scatter(k_means.centers_[center][0], k_means.centers_[center][1],c=colors[center], marker='*', s=150)

            for cat in k_means.clf_:
                for point in k_means.clf_[cat]:
                    plt.scatter(point[0], point[1], c=colors[cat])
            plt.show()
        std = np.std(cls_e_dis)
        stds.append(std)
        centers.append(k_means.centers_)
        cls.append(k_means.clf_)
    best_k = np.argmin(stds) + min_k
    best_centers = centers[np.argmin(stds)]
    best_cls = cls[np.argmin(stds)]
    print("best k:", best_k,"std:",np.min(stds))

    return best_centers,best_cls,best_k,np.min(stds)

def tradition_k_means(x,N,verbose=False):
    k_means = K_Means(k=N,only_dis=True)
    k_means.fit(x)
    cls_e_dis = []
    for c in k_means.clf_:
        cls_e = 0
        for feature in k_means.clf_[c]:
            for i, pos in enumerate(x):
                if (pos[0] == feature[0]).all():
                    cls_e += feature[1]
                    break
        cls_e_dis.append(cls_e)

    if verbose:
        # 绘制聚类过程的图
        plt.figure()
        plt.title(f"k={k}")
        for center in k_means.centers_:
            plt.scatter(k_means.centers_[center][0], k_means.centers_[center][1], c=colors[center], marker='*', s=150)

        for cat in k_means.clf_:
            for point in k_means.clf_[cat]:
                plt.scatter(point[0], point[1], c=colors[cat])
        plt.show()

    std = np.std(cls_e_dis)
    best_k = N
    best_centers = k_means.centers_
    best_cls = k_means.clf_
    print("k:", best_k, "std:", std)
    return best_centers,best_cls,best_k,std


def get_head(data,central_pos,only_dis=False):
    dis_matrix = []
    for i, sink_1 in enumerate(data):
        tmp = []
        for j, sink_2 in enumerate(data):
            dist_tp = np.linalg.norm(sink_1[0]-sink_2[0])
            tmp.append(dist_tp)
        dis_matrix.append(tmp)
    dis_matrix = np.array(dis_matrix)
    dist_record = []
    sink_number = len(data)
    for i,sink in enumerate(data):
        dis2center = np.linalg.norm(sink[0] - central_pos)
        if only_dis:
            tmp_dist = np.sum(dis_matrix[i, range(sink_number)])
        else:
            tmp_dist = 0.6*sink[1]-0.3*np.sum(dis_matrix[i, range(sink_number)]) - 0.1*dis2center
        dist_record.append(tmp_dist)
    central_node_index = np.argmax(dist_record)
    # central_sink = data[central_node_index]
    return central_node_index,dis_matrix

# 计算簇头到中央节点的路径
def get_head2center_path(cur_head_info,head_infos,central_pos):
    e_cost = []
    for i,head_info in enumerate(head_infos):
        if (head_info[0] == cur_head_info[0]).all():
            continue
        # 计算簇头间的距离和能耗
        dis_cls2cls = np.linalg.norm(cur_head_info[0]-head_info[0])
        cls2cls_pos_e = dis_cls2cls * dis_cls2cls * pos_data_size*data_press_ratio * pos_Eelec
        rec_e = pos_data_size*data_press_ratio * rec_Eelec
        if cur_head_info[1] < cls2cls_pos_e or head_info[1] < rec_e:
            continue
        # 计算跳转簇头到中央节点的距离和能耗
        dis_cls2ce = np.linalg.norm(head_info[0]-central_pos)
        pos_e = dis_cls2ce * dis_cls2ce * pos_data_size * data_press_ratio * pos_Eelec
        if pos_e+rec_e < head_info[1]:
            e_cost.append([cls2cls_pos_e,pos_e+rec_e,head_info[1],i])
    if e_cost:
        print(e_cost)
        tmp = [e[0]+e[1]-e[2] for e in e_cost]
        res_index = np.argmin(tmp)
        min_cost = e_cost[res_index]
        res_index = min_cost[3]
        return res_index,min_cost
    else:
        return [],[]






if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [7,2],[8, 8], [1, 0.6], [9, 11],[4,5],[6,2],[2,9]])
    e_mat = np.array([3,4,1,6,1,2,7,9,2,10])
    centers, cls, k = custom_k_means(x,max_k=6)

    for i,center in enumerate(centers):
        plt.scatter(centers[center][0], centers[center][1], c=colors[i],marker='*', s=150)

    for cat in cls:
        for point in cls[cat]:
            plt.scatter(point[0], point[1], c=colors[cat])

    # predict = [[2, 1], [6, 9]]
    # for feature in predict:
    #     cat = k_means.predict(predict)
    #     plt.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    plt.show()