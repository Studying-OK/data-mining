from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, ArrayType, TimestampType
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import matplotlib as cm
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import json

# --------------------------------
# 初始化Spark会话和全局配置
# --------------------------------
spark = SparkSession.builder \
    .appName("UserBehaviorAnalysis") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# --------------------------------
# 定义数据结构和全局变量
# --------------------------------
output_dir = "/home/wangyuhui/data mining"
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 14

last_login_schema = StructType([
    StructField("timestamps", TimestampType(), True)
])

login_schema = StructType([
    StructField("avg_session_duration", DoubleType(), True),
    StructField("devices", ArrayType(StringType()), True),
    StructField("first_login", StringType(), True),
    StructField("locations", ArrayType(StringType()), True),
    StructField("login_count", LongType(), True),
    StructField("timestamps", ArrayType(TimestampType()), True)
])

purchase_schema = StructType([
    StructField("avg_price", DoubleType(), True),
    StructField("categories", StringType(), True),
    StructField("items", ArrayType(StructType([
        StructField("id", LongType(), True)
    ])), True),
    StructField("payment_method", StringType(), True),
    StructField("payment_status", StringType(), True),
    StructField("purchase_date", StringType(), True)
])

preprocess_analysis = {
    "missing_values": {},
    "missing_percent": {},
}

analysis_results = {
    "user_profile": {
        "gender_dist": {},
        "age_dist": {"<18":0, "18-25":0, "25-35":0, "35-50":0, ">50":0},
        "always_login_dist": { "<18": {"0-6clock": 0, "6-12clock": 0, "12-18clock": 0, "18-24clock": 0},
            "18-25": {"0-6clock": 0, "6-12clock": 0, "12-18clock": 0, "18-24clock": 0},
            "25-35": {"0-6clock": 0, "6-12clock": 0, "12-18clock": 0, "18-24clock": 0},
            "35-50": {"0-6clock": 0, "6-12clock": 0, "12-18clock": 0, "18-24clock": 0},
            ">50": {"0-6clock": 0, "6-12clock": 0, "12-18clock": 0, "18-24clock": 0},},
        "country_dist": {},
        "total_users": 0  # 初始化为0，但需确保后续正确累加
    },
    "user_behavior": {
        "active_users": set(),
        "total_spent": 0.0,
    },
    "rfm_data": {}
}



# --------------------------------
# 数据预处理函数
# --------------------------------
def assess_data_quality(df):
    # 统计每一列的缺失值数量和比例
    total_rows = df.count()
    missing_values = {col: df.filter(F.col(col).isNull()).count() for col in df.columns}
    missing_percent = {col: (count / total_rows) * 100 for col, count in missing_values.items()}

    return  missing_values, missing_percent


def preprocess(df):
    try:
        # 1. 加载数据并验证
        partition_count = df.count()
        if partition_count == 0:
            raise ValueError("分区数据为空，无法进行处理")
        
        # 2. 数据清洗
        # 处理缺失值（示例：删除缺失值超过50%的列） 
        for col, percent in preprocess_analysis["missing_percent"].items():
            if percent > 50:
                df = df.drop(col)
                print(f"删除列 {col}，缺失比例超过50%")
        
    except Exception as e:
        print(f"预处理分区失败: {str(e)}")
          
# --------------------------------
# 核心处理函数
# --------------------------------
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
    current_time = time.time()  # 当前时间戳
    seconds_per_month = 30.44 * 24 * 60 * 60  # 将天转换为月，使用平均值 30.44 天/月
    rfm_df['recency'] = (current_time - rfm_df['last_purchase']) / seconds_per_month

    X = rfm_df[['recency', 'frequency', 'monetary']].copy()


    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # 将聚类结果添加到 DataFrame
    rfm_df['cluster'] = labels

    # 计算每个聚类的中心
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=['last_purchase', 'frequency', 'monetary'])

    # 构建保存结果
    clustering_results = {
        "cluster_centers": cluster_centers_df.to_dict(orient='records'),
        "user_clusters": rfm_df[['user_id', 'cluster']].to_dict(orient='records')
    }

    # 设置输出目录
    output_dir = "/home/wangyuhui/data mining"
    os.makedirs(output_dir, exist_ok=True)

    # 保存聚类结果到 JSON 文件
    clustering_output_file = os.path.join(output_dir, "clustering_results_10.json")
    with open(clustering_output_file, 'w') as f:
        json.dump(clustering_results, f, indent=4, default=str)

    print(f"聚类结果已保存到 {clustering_output_file}")

    important_points = []
    for cluster in rfm_df['cluster'].unique():
        cluster_data = rfm_df[rfm_df['cluster'] == cluster]
        cluster_center = cluster_data[['recency', 'frequency', 'monetary']].mean()
        cluster_data['distance'] = cluster_data.apply(lambda row: ((row['recency'] - cluster_center['recency'])**2 + 
                                                                (row['frequency'] - cluster_center['frequency'])**2 + 
                                                                (row['monetary'] - cluster_center['monetary'])**2)**0.5, axis=1)
        important_points.append(cluster_data.nsmallest(1000, 'distance'))  # 选择每个聚类中距离中心最近的1000个点

    important_points = pd.concat(important_points)


    # 可视化聚类结果（3D 散点图）
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 散点图
    scatter = ax.scatter(
        important_points['recency'],
        important_points['frequency'],
        important_points['monetary'],
        c=important_points['cluster'],
        cmap='viridis',
        s=10,
        marker='o'
    )

    # 设置标签和标题
    ax.set_xlabel('最近一次购买时间间隔（月）', fontsize=10)
    ax.set_ylabel('购买频率', fontsize=10)
    ax.set_zlabel('购买金额', fontsize=10)
    ax.set_title('10G数据集 RFM 三维聚类分析结果', fontsize=12)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(int(min(rfm_df['recency'])), max(rfm_df['recency']) + 1, 5))  # 根据实际数据范围调整
    ax.set_yticks(np.arange(int(min(rfm_df['frequency'])), max(rfm_df['frequency']) + 1, 1))
    ax.set_zticks(np.arange(int(min(rfm_df['monetary'])), max(rfm_df['monetary']) + 1, 500))

    # 添加图例
    legend = ax.legend(*scatter.legend_elements(), title="聚类")
    ax.add_artist(legend)

    # 保存可视化图表
    clustering_plot_file = os.path.join(output_dir, "rfm_clustering_3d_10.png")
    plt.savefig(clustering_plot_file)
    plt.close()

    print(f"聚类结果可视化已保存到 {clustering_plot_file}")


def process_partition(df):
    try:    
         # 3. 识别潜在高价值用户这次不写    
        df_parsed = df.withColumn(
            "purchase_data",
            F.from_json(F.col("purchase_history"), purchase_schema)
        ).filter(F.col("purchase_data").isNotNull())

        purchase_records = df_parsed.select(
            "id",
            F.col("purchase_data.purchase_date").alias("purchase_time"),
            F.size(F.col("purchase_data.items")).alias("frequency"),
            F.col("purchase_data.avg_price").alias("monetary")
        ).collect()
        
        for record in purchase_records:
            user_id = record["id"]
            if user_id not in analysis_results["rfm_data"]:
                analysis_results["rfm_data"][user_id] = {
                    "last_purchase": record["purchase_time"],
                    "frequency": record["frequency"],
                    "monetary": record["monetary"]
                }
            else:
                analysis_results["rfm_data"][user_id]["frequency"] += record["frequency"]
                analysis_results["rfm_data"][user_id]["monetary"] += record["monetary"]
                if record["purchase_time"] > analysis_results["rfm_data"][user_id]["last_purchase"]:
                    analysis_results["rfm_data"][user_id]["last_purchase"] = record["purchase_time"]
        
        # 释放内存
        df.unpersist()
        df_parsed.unpersist()
        
    except Exception as e:
        print(f"处理分区失败: {str(e)}")

# --------------------------------
# 主执行流程
# --------------------------------
if __name__ == "__main__":
    # 1. 获取所有分区文件（添加路径验证）
    data_dir = "/data2/wyh-dataset/10G_dataset/"
    partitions = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("part-")]
    # 记录开始时间
    start_time = time.time()
    for idx, partition_path in enumerate(partitions, 1):
        df = spark.read.parquet(partition_path)
        print(f"处理进度: {idx}/{len(partitions)} - {os.path.basename(partition_path)}")

        # 2. 数据预处理-评估数据质量
        missing_values, missing_percent = assess_data_quality(df)
        preprocess_analysis["missing_values"].update(missing_values)
        preprocess_analysis["missing_percent"].update(missing_percent)
        
        # 3. 数据预处理-数据加载
        preprocess(df)
        # 4. 处理用户行为数据
        process_partition(df)    
        
    perform_clustering(analysis_results["rfm_data"])
        

    print("length of analysis_results['rfm_data']:", len(analysis_results["rfm_data"]))
    end_time = time.time()
    print(f"处理完成，总耗时: {end_time - start_time:.2f}秒")
    summary_results = {
        "preprocess_analysis": preprocess_analysis,
        "analysis_results": analysis_results["rfm_data"],
        "time_taken": end_time - start_time,
    }
    # 将 summary_results 转换为 JSON 格式
    summary_json = json.dumps(summary_results, indent=4)

    # 定义输出文件路径
    summary_output_file = os.path.join(output_dir, "summary_results_10.json")

    # 将 JSON 数据写入文件
    with open(summary_output_file, 'w') as f:
        f.write(summary_json)
    spark.stop()