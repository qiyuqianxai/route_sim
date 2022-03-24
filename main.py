from algorithms import *
from model import *
from matplotlib import pyplot as plt
from settings import *

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

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
cost_e_record = []
surive_sink_record = []
active_sink_record = []
sink_avg_e_record = []
post_data_record = []
loss_data_record = []
ef_record = []
stds_record = []
sink_survival_cycles = []
vaild_e_record = []
for t in range(1,max_T):
    print("#" * 200)
    # 调用自定义的k-means
    N = max_k+1 if max_k < len(sink_infos) else len(sink_infos)+1
    if use_custom:
        _, cls, best_k,std = custom_k_means(sink_infos, N, verbose=False)
    else:

        N = kN if kN < len(sink_infos) else len(sink_infos)
        _, cls, best_k,std = tradition_k_means(sink_infos, N, verbose=False)
    stds_record.append(std)
    plt.figure()
    plt.title(f"第{t}轮聚类结果，存活sink数{len(sink_infos)}，best_k={best_k}")

    head_indexs = []
    cls_dis_mats = []
    head_info_list = []

    for cat in cls:
        # 选择簇头,得到距离矩阵
        if use_custom:
            head_index, cls_dis_mat = get_head(cls[cat],central_sink_pos)
        else:
            head_index, cls_dis_mat = get_head(cls[cat], central_sink_pos,True)
        head_info_list.append(cls[cat][head_index])
        head_indexs.append(head_index)
        cls_dis_mats.append(cls_dis_mat)
        for i,point in enumerate(cls[cat]):
            if i==head_index:
                plt.scatter(point[0][0], point[0][1], c=colors[cat], marker="^", s=400,alpha=point[1]/sink_init_e)
            else:
                plt.scatter(point[0][0], point[0][1], c=colors[cat],marker=".",s=350,alpha=point[1]/sink_init_e)
    plt.scatter([central_sink_pos[0]], [central_sink_pos[1]], marker="*", s=500, c="r", alpha=1)
    plt.savefig(f"{t}_res.png")
    plt.show()

    t_cost_E = 0
    vaild_e = 0
    t_pos_data = 0
    t_loss_data = 0
    active_sink_num = 0
    for head_index,cat,cls_dis_mat in zip(head_indexs,cls,cls_dis_mats):
        # sink节点产生数据
        flag = False
        for i,sink in enumerate(cls[cat]):
            tmp_e = 0
            # 无法产生数据，视为死亡
            if sink[1] < sink_gen_data_e:
                sink_survival_cycles.append(t)
                sink[1] = 0
                break
            # 每一轮sink都产生数据
            sink[1] -= sink_gen_data_e
            sink[2] = np.append(sink[2], np.random.randint(0, 2, pos_data_size))
            t_cost_E += sink_gen_data_e
            tmp_e += sink_gen_data_e

            # 计算该sink的发送能耗
            if i != head_index:
                # 计算能否顺利将数据发送到簇头
                dis = cls_dis_mat[i,head_index]
                pos_e = dis*dis*pos_data_size*pos_Eelec
                rec_e = pos_data_size * rec_Eelec
                if sink[1] < pos_e:
                    print("发送能量",pos_e)
                    continue
                else:
                    active_sink_num += 1
                    sink[1] -= pos_e
                    t_cost_E += pos_e
                    tmp_e += pos_e

                    print(f"簇{cat}，sink{i}缓存区数据：", sink[2])
                    pri_index = np.argwhere(sink[2] == 1)
                    if len(pri_index) >= pos_data_size:
                        sink_pos_data = np.squeeze(sink[2][pri_index][:pos_data_size])
                        sink[2] = np.delete(sink[2], pri_index[:pos_data_size])
                    else:
                        sink_pos_data = sink[2][pri_index]
                        sink[2] = np.delete(sink[2], pri_index)
                        sink_pos_data = np.append(sink_pos_data, [sink[2][:pos_data_size - len(pri_index)]])
                        sink[2] = np.delete(sink[2], range(pos_data_size - len(pri_index)))
                    print(f"簇{cat}，sink{i}发送包数据：", sink_pos_data)
                    print(f"簇{cat}，sink{i}缓存区剩余数据：", sink[2])
                # 计算能否接收
                if cls[cat][head_index][1] < rec_e:
                    # 丢包
                    t_loss_data += pos_data_size
                    continue
                else:
                    flag = True
                    cls[cat][head_index][1] -= rec_e
                    t_cost_E += rec_e
                    tmp_e += rec_e

            # 计算簇头的汇聚能耗+发送到中央节点的发送能耗
            merge_e = pos_data_size*E_dn
            # 若簇头无能量汇聚则丢包，
            if cls[cat][head_index][1] < merge_e:
                t_loss_data += pos_data_size
                continue
            else:
                # 若是汇聚成功，
                cls[cat][head_index][1] -= merge_e
                t_cost_E += merge_e
                tmp_e += merge_e

            # 汇聚后降低发送数据的尺寸
            dis2center = np.linalg.norm(cls[cat][head_index][0] - central_sink_pos)
            cls_head_e = dis2center * dis2center * pos_data_size * data_press_ratio * pos_Eelec

            if use_pri_path:
                res_index, e_cost = get_head2center_path(cls[cat][head_index], head_info_list, central_sink_pos)
                if e_cost:
                    if (cls_head_e > e_cost[0] + e_cost[1]) or (cls[cat][head_index][1] < cls_head_e):

                        cls[cat][head_index][1] -= e_cost[0]
                        cls[res_index][head_indexs[res_index]] -= e_cost[1]
                        t_cost_E += (e_cost[0] + e_cost[1])
                        tmp_e += (e_cost[0] + e_cost[1])
                    else:
                        cls[cat][head_index][1] -= cls_head_e
                        t_cost_E += cls_head_e
                        tmp_e += cls_head_e
                        # 数据顺利发送到中心视为成功
                    t_pos_data += pos_data_size
                    vaild_e += tmp_e
                    continue

            if cls[cat][head_index][1] > cls_head_e:
                cls[cat][head_index][1] -= cls_head_e
                t_cost_E += cls_head_e
                tmp_e += cls_head_e
                # 数据顺利发送到中心视为成功
                t_pos_data += pos_data_size
                vaild_e += tmp_e
            else:
                # 丢包
                t_loss_data += pos_data_size

        if flag:
            # 判断簇头是否活跃
            active_sink_num += 1

    # 更新sink_infos
    sink_infos = []
    avg_sink_e = 0
    for cat in cls:
        for sink in cls[cat]:
            # 死亡的sink放弃
            if sink[1] > 0:
                sink_infos.append(sink)
            avg_sink_e += sink[1]

    if len(sink_infos) > 0:
        avg_sink_e/=(total_sink_number-1)
        print(f"第{t}轮总耗能：{t_cost_E}")
        print(f"第{t}轮有效耗能：{vaild_e}")
        print(f"第{t}轮存活sink数：{len(sink_infos)}")
        print(f"第{t}轮活跃的sink数：{active_sink_num}")
        print(f"第{t}轮存活sink的平均能量：{avg_sink_e}")
        print(f"第{t}轮的发送成功的数据大小：{t_pos_data}")
        print(f"第{t}轮的发送失败的数据大小：{t_loss_data}")
        if (t_loss_data+t_pos_data) != 0:
            print(f"第{t}轮的丢包率为：{t_loss_data/(t_loss_data+t_pos_data)}")
        print("#"*200)
        cost_e_record.append(t_cost_E)
        vaild_e_record.append(vaild_e)
        surive_sink_record.append(len(sink_infos))
        active_sink_record.append(active_sink_num)
        sink_avg_e_record.append(avg_sink_e)
        post_data_record.append(t_pos_data)
        loss_data_record.append(t_loss_data)
        if (t_loss_data+t_pos_data) != 0:
            ef_record.append(t_loss_data/(t_loss_data+t_pos_data))
    else:
        # sink_survival_cycles += [t for i in sink_infos]
        # print(f"第{t}轮总耗能：{t_cost_E}")
        # print(f"第{t}轮存活sink数：{len(sink_infos)}")
        # print(f"第{t}轮活跃的sink数：{active_sink_num}")
        # print(f"第{t}轮的发送成功的数据大小：{t_pos_data}")
        # print(f"第{t}轮的发送失败的数据大小：{t_loss_data}")
        # if t_cost_E == 0:
        #     print(f"第{t}轮的丢包率为：0")
        # else:
        #     print(f"第{t}轮的丢包率为：{t_pos_data / t_cost_E}")
        print("#"*200)
        print(f"所有sink节点存活周期数：{sink_survival_cycles}")
        print(f"所有节点死亡，实验结束！节点平均存活周期{sum(sink_survival_cycles) / len(sink_survival_cycles)}")
        # cost_e_record.append(t_cost_E)
        # surive_sink_record.append(len(sink_infos))
        # active_sink_record.append(active_sink_num)
        # sink_avg_e_record.append(avg_sink_e)
        # post_data_record.append(t_pos_data)
        # loss_data_record.append(t_loss_data)
        # if t_cost_E == 0:
        #     ef_record.append(0)
        # else:
        #     ef_record.append(t_pos_data / t_cost_E)
        break
import datetime
import json
date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
save_name = "ck" if use_custom else "k{}".format(kN)
save_name = save_name+"_with_path_pri" if use_pri_path else save_name
with open(f"{save_name}.json","w",encoding="utf-8") as f:
    f.write(json.dumps({
        "每轮的总能耗":cost_e_record,
        "每轮的有效能耗":vaild_e_record,
        "每轮存活的sink数":surive_sink_record,
        "每轮存活的sink的平均能量":sink_avg_e_record,
        "每轮活跃的sink数":active_sink_record,
        "每轮发送成功的数据":post_data_record,
        "每轮发送失败的数据":loss_data_record,
        "每轮的丢包率":ef_record,
        "所有节点的存活周期":sink_survival_cycles,
        "每轮聚类后的各簇间的能量标准差":stds_record
    },indent=4,ensure_ascii=False))

import openpyxl
import xlwt
writebook = xlwt.Workbook()#打开一个excel
sheet = writebook.add_sheet("0")#在打开的excel中添加一个sheet
content = {
        "每轮的总能耗":cost_e_record,
        "每轮的有效能耗":vaild_e_record,
        "每轮存活的sink数":surive_sink_record,
        "每轮存活的sink的平均能量":sink_avg_e_record,
        "每轮活跃的sink数":active_sink_record,
        "每轮发送成功的数据":post_data_record,
        "每轮发送失败的数据":loss_data_record,
        "每轮的丢包率":ef_record,
        "所有节点的存活周期":sink_survival_cycles,
        "每轮聚类后的各簇间的能量标准差":stds_record
    }
for i,name in enumerate(content):
    # 写入数值名称
    sheet.write(0,i,name)
    # 写入值
    for j,value in enumerate(content[name]):
        sheet.write(j+1,i,value)
writebook.save(f'{save_name}.xls')#一定要记得保存















