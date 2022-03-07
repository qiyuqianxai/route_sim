############# 自定义参数 ##################
colors = ["b", "g", "r", "y", "m", "c", "w", "k"]
# 总的sink数
total_sink_number = 25
# sink初始能量
sink_init_e = 100
# sink 产生数据的最小能耗
sink_gen_data_e = 3

rec_Eelec = 0.161
pos_Eelec = 0.0397

E_dn = 0.05

pos_data_size = 3

# 汇聚后数据压缩比例
data_press_ratio = 0.2

# kmeans 最大k值
max_k = 6# max = 8
# 传统k-means的初始k值
kN = 4
# 运行周期数
max_T = 30

# 是否使用新的k-means
use_custom = True

use_pri_path = True