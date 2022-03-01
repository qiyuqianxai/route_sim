from kad_kmeans import *
from model import *
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

############# 自定义参数 ##################
colors = ["b", "g", "r", "y", "m", "c", "w", "k"]
# 总的sink数
total_sink_number = 20
# sink初始能量
sink_init_e = 300
# sink 产生数据的最小能耗
sink_gen_data_e = 5

rec_Eelec = 0.361
pos_Eelec = 0.197

E_dn = 0.05

pos_data_size = 3


# kmeans 最大k值
max_k = 6 # max = 8

# 运行周期数
max_T = 100

# 是否使用新的k-means
use_custom = False

same_seeds(2048)
##############################################

# 初始化模型
s_model = system_model(total_sink_number,sink_init_e)
s_model.init_nodes()
s_model.show_nodes()

central_sink_pos = s_model.nodes["central_sink"]
sinks = s_model.nodes["sinks"]
# sinks_pos = np.array([sink[:2] for sink in sinks])
# sinks_es = np.array([sink[2] for sink in sinks])
# 初始化数据
# print(sinks)
sink_infos = [sink+[(np.random.randint(0,2,pos_data_size))] for sink in sinks]
# print("sink_infos:",sink_infos)

# 模拟场景
sink_survival_cycles = []
for t in range(1,max_T):
    print("#" * 200)
    # 调用自定义的k-means
    N = max_k if max_k < len(sink_infos) else len(sink_infos)
    if use_custom:
        _, cls, best_k = custom_k_means(sink_infos, N, verbose=False)
    else:
        N = 2
        N = 3 if N < len(sink_infos) else len(sink_infos)
        _, cls, best_k = tradition_k_means(sink_infos, N, verbose=False)

    plt.figure()
    plt.title(f"第{t}轮聚类结果，存活sink数{len(sink_infos)}，best_k={best_k}")

    head_indexs = []
    cls_dis_mats = []
    for cat in cls:
        # 选择簇头,得到距离矩阵
        head_index, cls_dis_mat = get_head(cls[cat],central_sink_pos)
        head_indexs.append(head_index)
        cls_dis_mats.append(cls_dis_mat)
        for i,point in enumerate(cls[cat]):
            if i==head_index:
                plt.scatter(point[0][0], point[0][1], c=colors[cat], marker=".", s=point[1] * 20,alpha=1)
            else:
                plt.scatter(point[0][0], point[0][1], c=colors[cat],marker=".",s=point[1]*20,alpha=0.2)
    plt.scatter([central_sink_pos[0]], [central_sink_pos[1]], marker="*", s=300, c="r", alpha=1)
    plt.savefig(f"{t}_res.png")
    plt.show()


    t_cost_E = 0
    t_pos_data = 0
    t_loss_data = 0
    for head_index,cat,cls_dis_mat in zip(head_indexs,cls,cls_dis_mats):
        # sink节点产生数据
        for i,sink in enumerate(cls[cat]):
            # 无法产生数据，视为死亡
            if sink[1] < sink_gen_data_e:
                sink_survival_cycles.append(t)
                sink[1] = 0
                break
            # 每一轮sink都产生数据
            sink[2] = np.append(sink[2], np.random.randint(0, 2, pos_data_size))
            t_cost_E += sink_gen_data_e

            print(f"簇{cat}，sink{i}缓存区数据：",sink[2])
            pri_index = np.argwhere(sink[2]==1)
            if len(pri_index) >= pos_data_size:
                sink_pos_data = np.squeeze(sink[2][pri_index][:pos_data_size])
                sink[2] = np.delete(sink[2], pri_index[:pos_data_size])
            else:
                sink_pos_data = sink[2][pri_index]
                sink[2] = np.delete(sink[2], pri_index)
                sink_pos_data = np.append(sink_pos_data,[sink[2][:pos_data_size-len(pri_index)]])
                sink[2] = np.delete(sink[2], range(pos_data_size-len(pri_index)))
            print(f"簇{cat}，sink{i}发送包数据：", sink_pos_data)
            print(f"簇{cat}，sink{i}缓存区剩余数据：", sink[2])

            # 计算该sink的发送能耗
            if i != head_index:
                # 计算能否顺利将数据发送到簇头
                dis = cls_dis_mat[i,head_index]
                pos_e = dis*dis*pos_data_size*pos_Eelec
                rec_e = pos_data_size * rec_Eelec
                if sink[1] < pos_e:
                    continue
                elif cls[cat][head_index][1] < rec_e:
                    sink[1] -= pos_e
                    # 丢包
                    t_loss_data += pos_data_size
                    t_cost_E += pos_e
                    continue
                else:
                    cls[cat][head_index][1] -= rec_e
                    t_cost_E += rec_e

                # 计算簇头的接受能耗+汇聚能耗+发送到中央节点
                merge_e = pos_data_size*E_dn
                dis2center = np.linalg.norm(cls[cat][head_index][0] - central_sink_pos)
                cls_head_e = dis2center * dis2center * pos_data_size * pos_Eelec
                # 若簇头无能量接受和汇聚则丢包，
                if cls[cat][head_index][1] < (merge_e+cls_head_e):
                    t_loss_data += pos_data_size
                    continue
                else:
                    cls[cat][head_index][1] -= (merge_e+cls_head_e)
                    t_cost_E += (rec_e + merge_e)

                # 数据顺利发送到中心视为成功
                t_pos_data += pos_data_size

    # 更新sink_infos
    sink_infos = []
    avg_sink_e = 0
    for cat in cls:
        for sink in cls[cat]:
            # 死亡的sink放弃
            if sink[1] > 0:
                sink_infos.append(sink)
                avg_sink_e += sink[1]

    if len(sink_infos) > 1 and t_pos_data > 0:
        avg_sink_e/=len(sink_infos)
        print(f"第{t}轮总耗能：{t_cost_E}")
        print(f"第{t}轮存活sink数：{len(sink_infos)}")
        print(f"第{t}轮存活sink的平均能量：{avg_sink_e}")
        print(f"第{t}轮的发送成功的数据大小：{t_pos_data}")
        print(f"第{t}轮的发送失败的数据大小：{t_loss_data}")
        print(f"第{t}轮的能效为：{t_pos_data/t_cost_E}")
        print("#"*200)
    else:
        sink_survival_cycles += [t for i in sink_infos]
        print(f"第{t}轮总耗能：{t_cost_E}")
        print(f"第{t}轮存活sink数：{len(sink_infos)}")
        print(f"第{t}轮的发送成功的数据大小：{t_pos_data}")
        print(f"第{t}轮的发送失败的数据大小：{t_loss_data}")
        print(f"第{t}轮的能效为：{t_pos_data / t_cost_E}")
        print("#"*200)
        print(f"所有sink节点存活周期数：{sink_survival_cycles}")
        print(f"整个网络无法正常运行，实验结束！节点平均存活周期{sum(sink_survival_cycles) / len(sink_survival_cycles)}")
        break



















