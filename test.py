import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 14

# 示例 RFM 数据
rfm_data = {
    1: {
        "last_purchase": "2023-10-05",
        "frequency": 5,
        "monetary": 1500.0
    },
    2: {
        "last_purchase": "2023-12-20",
        "frequency": 3,
        "monetary": 900.0
    },
    3: {
        "last_purchase": "2023-08-15",
        "frequency": 8,
        "monetary": 2200.0
    },
    4: {
        "last_purchase": "2023-11-30",
        "frequency": 2,
        "monetary": 600.0
    },
    5: {
        "last_purchase": "2023-09-22",
        "frequency": 4,
        "monetary": 1200.0
    }
}

# 定义 KMeans 聚类函数
def perform_clustering(rfm_data):
    if not rfm_data:
        print("RFM 数据为空，跳过聚类分析。")
        return

    # 将字典转换为 Pandas DataFrame
    rfm_df = pd.DataFrame.from_dict(rfm_data, orient='index')
    rfm_df.reset_index(inplace=True)
    rfm_df.rename(columns={'index': 'user_id'}, inplace=True)

    # 将 last_purchase 字段转换为时间戳
    rfm_df['last_purchase'] = rfm_df['last_purchase'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())

    # 计算最近一次购买时间间隔（以月为单位）
    current_time = time.time()
    seconds_per_month = 30.44 * 24 * 60 * 60
    rfm_df['recency'] = (current_time - rfm_df['last_purchase']) / seconds_per_month
    print("数据预处理前")
    print(rfm_df["last_purchase"])
    print("数据预处理后")
    print(rfm_df['recency'])


#     # 特征选择
#     X = rfm_df[['recency', 'frequency', 'monetary']].copy()

#     # 数据标准化
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # KMeans 聚类
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     kmeans.fit(X_scaled)
#     labels = kmeans.labels_
#     rfm_df['cluster'] = labels

#     # 计算聚类中心
#     cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
#     cluster_centers_df = pd.DataFrame(cluster_centers, columns=['recency', 'frequency', 'monetary'])

#     # 构建保存结果
#     clustering_results = {
#         "cluster_centers": cluster_centers_df.to_dict(orient='records'),
#         "user_clusters": rfm_df[['user_id', 'cluster']].to_dict(orient='records')
#     }

#     # 设置输出目录
#     output_dir = "/home/wangyuhui/data mining"
#     os.makedirs(output_dir, exist_ok=True)

#     # 保存聚类结果到 JSON 文件
#     clustering_output_file = os.path.join(output_dir, "clustering_results.json")
#     with open(clustering_output_file, 'w') as f:
#         json.dump(clustering_results, f, indent=4, default=str)

#     print(f"聚类结果已保存到 {clustering_output_file}")


#     # 可视化聚类结果（3D 散点图）
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制 3D 散点图
#     scatter = ax.scatter(
#         rfm_df['recency'],
#         rfm_df['frequency'],
#         rfm_df['monetary'],
#         c=rfm_df['cluster'],
#         cmap='viridis',
#         s=100,
#         marker='o'
#     )

#     # 设置标签和标题
#     ax.set_xlabel('最近一次购买时间间隔（月）', fontsize=10)
#     ax.set_ylabel('购买频率', fontsize=10)
#     ax.set_zlabel('购买金额', fontsize=10)
#     ax.set_title('RFM 三维聚类分析结果', fontsize=12)

#     # 设置坐标轴刻度
#     ax.set_xticks(np.arange(int(min(rfm_df['recency'])), max(rfm_df['recency']) + 1, 1))  # 根据实际数据范围调整
#     ax.set_yticks(np.arange(int(min(rfm_df['frequency'])), max(rfm_df['frequency']) + 1, 1))
#     ax.set_zticks(np.arange(int(min(rfm_df['monetary'])), max(rfm_df['monetary']) + 1, 500))

#     # 添加图例
#     legend = ax.legend(*scatter.legend_elements(), title="聚类")
#     ax.add_artist(legend)

#     # 保存可视化图表
#     clustering_plot_file = os.path.join(output_dir, "rfm_clustering_3d.png")
#     plt.savefig(clustering_plot_file)
#     plt.close()

#     print(f"聚类结果可视化已保存到 {clustering_plot_file}")

# # 调用聚类函数
perform_clustering(rfm_data)