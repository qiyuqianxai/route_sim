import json
import os
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
with open("ck.json","r",encoding="utf-8")as f:
    ck = json.load(f)
    ck_cost_e_record = ck["每轮的总能耗"]
    ck_vaild_e_record = ck["每轮的有效能耗"]
    ck_surive_sink_record = ck["每轮存活的sink数"]
    ck_sink_avg_e_record = ck["每轮存活的sink的平均能量"]
    ck_active_sink_record = ck["每轮活跃的sink数"]
    ck_post_data_record = ck["每轮发送成功的数据"]
    ck_loss_data_record = ck["每轮发送失败的数据"]
    ck_ef_record = ck["每轮的丢包率"]
    ck_sink_survival_cycles = ck["所有节点的存活周期"]
    ck_stds_record = ck["每轮聚类后的各簇间的能量标准差"]

with open("ck_with_path_pri.json","r",encoding="utf-8")as f:
    ckp = json.load(f)
    ckp_cost_e_record = ckp["每轮的总能耗"]
    ckp_vaild_e_record = ckp["每轮的有效能耗"]
    ckp_surive_sink_record = ckp["每轮存活的sink数"]
    ckp_sink_avg_e_record = ckp["每轮存活的sink的平均能量"]
    ckp_active_sink_record = ckp["每轮活跃的sink数"]
    ckp_post_data_record = ckp["每轮发送成功的数据"]
    ckp_loss_data_record = ckp["每轮发送失败的数据"]
    ckp_ef_record = ckp["每轮的丢包率"]
    ckp_sink_survival_cycles = ckp["所有节点的存活周期"]
    ckp_stds_record = ckp["每轮聚类后的各簇间的能量标准差"]

with open("k2.json","r",encoding="utf-8")as f:
    k2 = json.load(f)
    k2_cost_e_record = k2["每轮的总能耗"]
    k2_vaild_e_record = k2["每轮的有效能耗"]
    k2_surive_sink_record = k2["每轮存活的sink数"]
    k2_sink_avg_e_record = k2["每轮存活的sink的平均能量"]
    k2_active_sink_record = k2["每轮活跃的sink数"]
    k2_post_data_record = k2["每轮发送成功的数据"]
    k2_loss_data_record = k2["每轮发送失败的数据"]
    k2_ef_record = k2["每轮的丢包率"]
    k2_sink_survival_cycles = k2["所有节点的存活周期"]
    k2_stds_record = k2["每轮聚类后的各簇间的能量标准差"]

with open("k3.json","r",encoding="utf-8")as f:
    k3 = json.load(f)
    k3_cost_e_record = k3["每轮的总能耗"]
    k3_vaild_e_record = k3["每轮的有效能耗"]
    k3_surive_sink_record = k3["每轮存活的sink数"]
    k3_sink_avg_e_record = k3["每轮存活的sink的平均能量"]
    k3_active_sink_record = k3["每轮活跃的sink数"]
    k3_post_data_record = k3["每轮发送成功的数据"]
    k3_loss_data_record = k3["每轮发送失败的数据"]
    k3_ef_record = k3["每轮的丢包率"]
    k3_sink_survival_cycles = k3["所有节点的存活周期"]
    k3_stds_record = k3["每轮聚类后的各簇间的能量标准差"]

with open("k4.json","r",encoding="utf-8")as f:
    k4 = json.load(f)
    k4_cost_e_record = k4["每轮的总能耗"]
    k4_vaild_e_record = k4["每轮的有效能耗"]
    k4_surive_sink_record = k4["每轮存活的sink数"]
    k4_sink_avg_e_record = k4["每轮存活的sink的平均能量"]
    k4_active_sink_record = k4["每轮活跃的sink数"]
    k4_post_data_record = k4["每轮发送成功的数据"]
    k4_loss_data_record = k4["每轮发送失败的数据"]
    k4_ef_record = k4["每轮的丢包率"]
    k4_sink_survival_cycles = k4["所有节点的存活周期"]
    k4_stds_record = k4["每轮聚类后的各簇间的能量标准差"]


plt.figure()
plt.title("有效能耗对比")
plt.plot(np.array(ck_vaild_e_record), c='r', label='ck-每轮的有效能耗')
plt.plot(np.array(k2_vaild_e_record), c='b', label='k2-每轮的有效能耗')
plt.plot(np.array(k3_vaild_e_record), c='g', label='k3-每轮的有效能耗')
plt.plot(np.array(k4_vaild_e_record), c='y', label='k4-每轮的有效能耗')
plt.plot(np.array(ckp_vaild_e_record), c='m', label='ckp-每轮的有效能耗')
plt.legend(loc='best')
plt.ylabel('能耗值')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—cost_e.png")

plt.figure()
plt.title("能效对比")
val = [np.sum(np.array(ck_vaild_e_record))/np.sum(np.array(ck_cost_e_record)),
       np.sum(np.array(k2_vaild_e_record))/np.sum(np.array(k2_cost_e_record)),
       np.sum(np.array(k3_vaild_e_record))/np.sum(np.array(k3_cost_e_record)),
       np.sum(np.array(k4_vaild_e_record))/np.sum(np.array(k4_cost_e_record)),
       np.sum(np.array(ckp_vaild_e_record))/np.sum(np.array(ckp_cost_e_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.8)  # or `color=['r', 'g', 'b']`

plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.ylabel('能效比')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—cost_e_avg.png")

plt.figure()
plt.title("活跃的sink数对比")
plt.plot(np.array(ck_active_sink_record), c='r', label='ck-每轮活跃的sink数')
plt.plot(np.array(k2_active_sink_record), c='b', label='k2-每轮活跃的sink数')
plt.plot(np.array(k3_active_sink_record), c='g', label='k3-每轮活跃的sink数')
plt.plot(np.array(k4_active_sink_record), c='y', label='k4-每轮活跃的sink数')
plt.plot(np.array(ckp_active_sink_record), c='m', label='ckp-每轮活跃的sink数')
plt.legend(loc='best')
plt.ylabel('活跃的sink数')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—active_sink.png")

plt.figure()
plt.title("活跃的sink数对比")
val = [np.max(np.array(ck_active_sink_record)),np.max(np.array(k2_active_sink_record)),
       np.max(np.array(k3_active_sink_record)),np.max(np.array(k4_active_sink_record)),
       np.max(np.array(ckp_active_sink_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.2)  # or `color=['r', 'g', 'b']`

val = [np.average(np.array(ck_active_sink_record)),np.average(np.array(k2_active_sink_record)),
       np.average(np.array(k3_active_sink_record)),np.average(np.array(k4_active_sink_record)),
       np.average(np.array(ckp_active_sink_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.6)  # or `color=['r', 'g', 'b']`

val = [np.min(np.array(ck_active_sink_record)),np.min(np.array(k2_active_sink_record)),
       np.min(np.array(k3_active_sink_record)),np.min(np.array(k4_active_sink_record)),
       np.min(np.array(ckp_active_sink_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`

plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.ylabel('活跃的sink数')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—active_sink_avg.png")

plt.figure()
plt.title("各簇间的能量标准差对比")
plt.plot(np.array(ck_stds_record), c='r', label='ck-每轮聚类后的各簇间的能量标准差')
plt.plot(np.array(k2_stds_record), c='b', label='k2-每轮聚类后的各簇间的能量标准差')
plt.plot(np.array(k3_stds_record), c='g', label='k3-每轮聚类后的各簇间的能量标准差')
plt.plot(np.array(k4_stds_record), c='y', label='k4-每轮聚类后的各簇间的能量标准差')
plt.plot(np.array(ckp_stds_record), c='m', label='ckp-每轮聚类后的各簇间的能量标准差')
plt.legend(loc='best')
plt.ylabel('能量标准差')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—std.png")

plt.figure()
plt.title("各簇间的能量标准差对比")
val = [np.max(np.array(ck_stds_record)),np.max(np.array(k2_stds_record)),
       np.max(np.array(k3_stds_record)),np.max(np.array(k4_stds_record)),
       np.max(np.array(ckp_stds_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.2)  # or `color=['r', 'g', 'b']`

val = [np.average(np.array(ck_stds_record)),np.average(np.array(k2_stds_record)),
       np.average(np.array(k3_stds_record)),np.average(np.array(k4_stds_record)),
       np.average(np.array(ckp_stds_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.6)  # or `color=['r', 'g', 'b']`

val = [np.min(np.array(ck_stds_record)),np.min(np.array(k2_stds_record)),
       np.min(np.array(k3_stds_record)),np.min(np.array(k4_stds_record)),
       np.min(np.array(ckp_stds_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.ylabel('能量标准差')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—std_avg.png")

plt.figure()
plt.title("丢包率对比")
plt.plot(np.array(ck_ef_record), c='r', label='ck-每轮的丢包率')
plt.plot(np.array(k2_ef_record), c='b', label='k2-每轮的丢包率')
plt.plot(np.array(k3_ef_record), c='g', label='k3-每轮的丢包率')
plt.plot(np.array(k4_ef_record), c='y', label='k4-每轮的丢包率')
plt.plot(np.array(ckp_ef_record), c='m', label='ckp-每轮的丢包率')
plt.legend(loc='best')
plt.ylabel('丢包率值')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—ef.png")

plt.figure()
plt.title("丢包率对比")
val = [np.max(np.array(ck_ef_record)),np.max(np.array(k2_ef_record)),
       np.max(np.array(k3_ef_record)),np.max(np.array(k4_ef_record)),
       np.max(np.array(ckp_ef_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.3)  # or `color=['r', 'g', 'b']`

val = [np.average(np.array(ck_ef_record)),np.average(np.array(k2_ef_record)),
       np.average(np.array(k3_ef_record)),np.average(np.array(k4_ef_record)),
       np.average(np.array(ckp_ef_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.8)  # or `color=['r', 'g', 'b']`

val = [np.min(np.array(ck_ef_record)),np.min(np.array(k2_ef_record)),
       np.min(np.array(k3_ef_record)),np.min(np.array(k4_ef_record)),
       np.min(np.array(ckp_ef_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`

plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.ylabel('丢包率')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—ef_avg.png")

plt.figure()
plt.title("发送成功的数据量对比")
plt.plot(np.array(ck_post_data_record), c='r', label='ck-每轮发送成功的数据')
plt.plot(np.array(k2_post_data_record), c='b', label='k2-每轮发送成功的数据')
plt.plot(np.array(k3_post_data_record), c='g', label='k3-每轮发送成功的数据')
plt.plot(np.array(k4_post_data_record), c='y', label='k4-每轮发送成功的数据')
plt.plot(np.array(ckp_post_data_record), c='m', label='ckp-每轮发送成功的数据')
plt.legend(loc='best')
plt.ylabel('数据包')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—pos_data.png")

plt.figure()
plt.title("发送成功的数据量对比")

val = [np.sum(np.array(ck_post_data_record)),np.sum(np.array(k2_post_data_record)),
       np.sum(np.array(k3_post_data_record)),np.sum(np.array(k4_post_data_record)),
       np.sum(np.array(ckp_post_data_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.8)  # or `color=['r', 'g', 'b']`

# val = [np.average(np.array(ck_post_data_record)),np.average(np.array(k2_post_data_record)),
#        np.average(np.array(k3_post_data_record)),np.average(np.array(k4_post_data_record)),
#        np.average(np.array(ckp_post_data_record))]
# plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.6)  # or `color=['r', 'g', 'b']`
#
# val = [np.min(np.array(ck_post_data_record)),np.min(np.array(k2_post_data_record)),
#        np.min(np.array(k3_post_data_record)),np.min(np.array(k4_post_data_record)),
#        np.min(np.array(ckp_post_data_record))]
# plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)

plt.ylabel('发送成功的数据量')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—pos_data_avg.png")

plt.figure()
plt.title("发送失败的数据量对比")
plt.plot(np.array(ck_loss_data_record), c='r', label='ck-每轮发送失败的数据')
plt.plot(np.array(k2_loss_data_record), c='b', label='k2-每轮发送失败的数据')
plt.plot(np.array(k3_loss_data_record), c='g', label='k3-每轮发送失败的数据')
plt.plot(np.array(k4_loss_data_record), c='y', label='k4-每轮发送失败的数据')
plt.plot(np.array(ckp_loss_data_record), c='m', label='ckp-每轮发送失败的数据')
plt.legend(loc='best')
plt.ylabel('数据包')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—loss_data.png")

plt.figure()
plt.title("发送失败的数据量对比")
val = [np.sum(np.array(ck_loss_data_record)),np.sum(np.array(k2_loss_data_record)),
       np.sum(np.array(k3_loss_data_record)),np.sum(np.array(k4_loss_data_record)),
       np.sum(np.array(ckp_loss_data_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.8)  # or `color=['r', 'g', 'b']`

# val = [np.average(np.array(ck_loss_data_record)),np.average(np.array(k2_loss_data_record)),
#        np.average(np.array(k3_loss_data_record)),np.average(np.array(k4_loss_data_record)),
#        np.average(np.array(ckp_loss_data_record))]
# plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.6)  # or `color=['r', 'g', 'b']`
#
# val = [np.min(np.array(ck_loss_data_record)),np.min(np.array(k2_loss_data_record)),
#        np.min(np.array(k3_loss_data_record)),np.min(np.array(k4_loss_data_record)),
#        np.min(np.array(ckp_loss_data_record))]
#
# plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)

plt.ylabel('发送失败的数据量')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—loss_data_avg.png")



plt.figure()
plt.title("节点的存活周期")
val = [np.max(np.array(ck_sink_survival_cycles)),np.max(np.array(k2_sink_survival_cycles)),
       np.max(np.array(k3_sink_survival_cycles)),np.max(np.array(k4_sink_survival_cycles)),
       np.max(np.array(ckp_sink_survival_cycles))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.2)  # or `color=['r', 'g', 'b']`

val = [np.average(np.array(ck_sink_survival_cycles)),np.average(np.array(k2_sink_survival_cycles)),
       np.average(np.array(k3_sink_survival_cycles)),np.average(np.array(k4_sink_survival_cycles)),
       np.average(np.array(ckp_sink_survival_cycles))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.6)  # or `color=['r', 'g', 'b']`

val = [np.min(np.array(ck_sink_survival_cycles)),np.min(np.array(k2_sink_survival_cycles)),
       np.min(np.array(k3_sink_survival_cycles)),np.min(np.array(k4_sink_survival_cycles)),
       np.min(np.array(ckp_sink_survival_cycles))]

plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.ylabel('节点存活周期')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—sink_survival_cycles_avg.png")


plt.figure()
plt.title("每轮的sink剩余能量均值对比")
plt.plot(np.array(ck_sink_avg_e_record), c='r', label='ck-每轮的sink剩余能量均值')
plt.plot(np.array(k2_sink_avg_e_record), c='b', label='k2-每轮的sink剩余能量均值')
plt.plot(np.array(k3_sink_avg_e_record), c='g', label='k3-每轮的sink剩余能量均值')
plt.plot(np.array(k4_sink_avg_e_record), c='y', label='k4-每轮的sink剩余能量均值')
plt.plot(np.array(ckp_sink_avg_e_record), c='m', label='ckp-每轮的sink剩余能量均值')
plt.legend(loc='best')
plt.ylabel('活跃的sink数')
plt.xlabel('周期')
plt.grid()
plt.savefig("contrast—sink_avg_e.png")

plt.figure()
plt.title("sink剩余能量均值对比")
val = [np.max(np.array(ck_sink_avg_e_record)),np.max(np.array(k2_sink_avg_e_record)),
       np.max(np.array(k3_sink_avg_e_record)),np.max(np.array(k4_sink_avg_e_record)),
       np.max(np.array(ckp_sink_avg_e_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.2)  # or `color=['r', 'g', 'b']`

val = [np.average(np.array(ck_sink_avg_e_record)),np.average(np.array(k2_sink_avg_e_record)),
       np.average(np.array(k3_sink_avg_e_record)),np.average(np.array(k4_sink_avg_e_record)),
       np.average(np.array(ckp_sink_avg_e_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=0.6)  # or `color=['r', 'g', 'b']`

val = [np.min(np.array(ck_sink_avg_e_record)),np.min(np.array(k2_sink_avg_e_record)),
       np.min(np.array(k3_sink_avg_e_record)),np.min(np.array(k4_sink_avg_e_record)),
       np.min(np.array(ckp_sink_avg_e_record))]
plt.bar(["ck","k2","k3","k4","ckp"], val, color='rbgym',alpha=1)  # or `color=['r', 'g', 'b']`

plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.ylabel('sink剩余能量均值对比')
plt.xlabel('methods')
# plt.grid()
plt.savefig("contrast—sink_avg_e2.png")