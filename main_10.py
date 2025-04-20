from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, ArrayType, TimestampType
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
        "always_login_dist": {"0-6clock": 0, "6-12clock": 0, "12-18clock": 0, "18-24clock": 0},
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
# 设置可视化风格
# --------------------------------
# 设置可视化风格
plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({'font.size': 12})

# 创建输出目录
analyse_dir = "/home/wangyuhui/data mining"
output_dir = os.path.join(analyse_dir, "visualization_10")
os.makedirs(output_dir, exist_ok=True)

# 性别分布（柱状图）
def plot_gender_distribution():
    gender_dist = analysis_results["user_profile"]["gender_dist"]
    
    if gender_dist:
        labels = list(gender_dist.keys())
        sizes = list(gender_dist.values())
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=sizes, palette='viridis')
        plt.title('用户性别分布')
        plt.xlabel('性别')
        plt.ylabel('用户数量')
        
        output_path = os.path.join(output_dir, "gender_distribution.png")
        plt.savefig(output_path)
        plt.close()
    else:
        print("性别分布数据为空，跳过可视化。")

# 年龄分布（饼图）
def plot_age_distribution():
    age_dist = analysis_results["user_profile"]["age_dist"]
    
    if sum(age_dist.values()) > 0:
        labels = list(age_dist.keys())
        sizes = list(age_dist.values())
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('用户年龄分布')
        plt.axis('equal')
        
        output_path = os.path.join(output_dir, "age_distribution.png")
        plt.savefig(output_path)
        plt.close()
    else:
        print("年龄分布数据为空，跳过可视化。")

# 登录时间分布（热力图）
def plot_login_time_distribution():
    login_dist = analysis_results["user_profile"]["always_login_dist"]
    
    if sum(login_dist.values()) > 0:
        time_periods = list(login_dist.keys())
        login_counts = list(login_dist.values())
        
        # 创建热力图数据
        data = [[count] for count in login_counts]
        
        plt.figure(figsize=(12, 7))
        sns.heatmap(data, annot=True, fmt='d', yticklabels=time_periods, cmap='YlGnBu')
        plt.title('用户登录时间分布')
        plt.xlabel('登录次数')
        plt.ylabel('时间段')
        
        output_path = os.path.join(output_dir, "login_time_distribution.png")
        plt.savefig(output_path)
        plt.close()
    else:
        print("登录时间分布数据为空，跳过可视化。")

# 国家分布（柱状图）
def plot_country_distribution(country_dist):
    if country_dist:
        labels = list(country_dist.keys())
        sizes = list(country_dist.values())
        total_users = sum(sizes)
        normalized_sizes = [(count / total_users) * 100 for count in sizes]  # 归一化为百分比
        
        plt.figure(figsize=(12, 7))
        sns.barplot(x=labels, y=normalized_sizes, palette='husl')
        plt.title('用户国家分布')
        plt.xlabel('国家')
        plt.ylabel('用户比例 (%)')
        plt.xticks(rotation=45)
        
        # 在柱子上显示具体百分比数值
        for index, value in enumerate(normalized_sizes):
            plt.text(index, value + 0.5, f'{value:.1f}%', ha='center')
        
        output_path = os.path.join(output_dir, "country_distribution.png")
        plt.savefig(output_path)
        plt.close()

# # 总消费金额（文本图）
# def plot_total_spent():
#     total_spent = analysis_results["user_behavior"]["total_spent"]/analysis_results["user_profile"]["total_users"]
    
#     if total_spent > 0:
#         plt.figure(figsize=(8, 5))
#         plt.text(0.5, 0.5, f'总消费金额: {total_spent:.2f}', fontsize=40, ha='center', va='center', color='blue')
#         plt.axis('off')
#         plt.title('总消费金额')
        
#         output_path = os.path.join(output_dir, "total_spent.png")
#         plt.savefig(output_path)
#         plt.close()
#     else:
#         print("总消费金额数据为空，跳过可视化。")


# 总用户数
def plot_total_users():
    total_users = analysis_results["user_profile"]["total_users"]
    
    if total_users > 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, f'总用户数: {total_users}', fontsize=40, ha='center', va='center', color='blue')
        plt.axis('off')
        plt.title('总用户数')
        
        output_path = os.path.join(output_dir, "total_users.png")
        plt.savefig(output_path)
        plt.close()
    else:
        print("总用户数数据为空，跳过可视化。")


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
def process_partition(df):
    try:
        # 1. 用户画像分析
        # 累加总用户数
        analysis_results["user_profile"]["total_users"] += df.count()
        
        # 性别分布
        gender_counts = df.groupBy("gender").count().collect()
        for row in gender_counts:
            gender = row["gender"]
            count = row["count"]
            analysis_results["user_profile"]["gender_dist"][gender] = \
                analysis_results["user_profile"]["gender_dist"].get(gender, 0) + count
       
        
        # 年龄分布分箱
        # 初始化年龄分布字典（确保键名与后续生成一致）
        analysis_results["user_profile"]["age_dist"] = {
            "<18": 0,
            "18-25": 0,
            "25-35": 0,
            "35-50": 0,
            ">50": 0
        }

        # 使用链式when表达式直接分箱
        df_age = df.withColumn(
            "age_group",
            F.when(F.col("age") < 18, "<18")
            .when((F.col("age") >= 18) & (F.col("age") < 25), "18-25")
            .when((F.col("age") >= 25) & (F.col("age") < 35), "25-35")
            .when((F.col("age") >= 35) & (F.col("age") < 50), "35-50")
            .otherwise(">50")  # 明确处理所有>=50的情况
        )
        age_counts = df_age.groupBy("age_group").count().collect()
        for row in age_counts:
            age_group = row["age_group"]
            count = row["count"]
            analysis_results["user_profile"]["age_dist"][age_group] += count
            
                
        # 国家分布Top10
        country_counts = df.groupBy("country").count().orderBy(F.desc("count")).limit(10).collect()
        for row in country_counts:
            country = row["country"]
            count = row["count"]
            analysis_results["user_profile"]["country_dist"][country] = \
                analysis_results["user_profile"]["country_dist"].get(country, 0) + count
        
        # 用户登录时间分布（0-6点、6-12点、12-18点、18-24点）
        df_parsed = df.withColumn(
            "login_data",
            F.from_json(F.col("login_history"), login_schema)
        )
        
        df_login_times = df_parsed.selectExpr("id as user_id", "login_data.timestamps as login_timestamps")
        df_exploded = df_login_times.selectExpr("user_id", "explode(login_timestamps) as login_time")
        df_exploded = df_exploded.withColumn("hour", F.hour("login_time"))

        # 定义时间段映射
        time_period_mapping = {
            "0-6clock": (0, 6),
            "6-12clock": (6, 12),
            "12-18clock": (12, 18),
            "18-24clock": (18, 24)
        }
        for period, (start, end) in time_period_mapping.items():
            count = df_exploded.filter((F.col("hour") >= start) & (F.col("hour") < end)).count()
            analysis_results["user_profile"]["always_login_dist"][period] += count


        
        # 2. 用户行为分析
        # 解析购买记录
        df_parsed = df.withColumn(
            "purchase_data",
            F.from_json(F.col("purchase_history"), purchase_schema)
        ).filter(F.col("purchase_data").isNotNull())
        
        # 活跃用户（最近30天登录）
        active_users = df_parsed.filter(F.col("is_active") == True).select("id").rdd.flatMap(lambda x: x).collect()
        analysis_results["user_behavior"]["active_users"].update(active_users)
        
        total_spent = df_parsed.select(F.sum("purchase_data.avg_price")).collect()[0][0]
        if total_spent is not None:
            analysis_results["user_behavior"]["total_spent"] += total_spent
        

        
        # # 3. 识别潜在高价值用户这次不写
        # purchase_records = df_parsed.select(
        #     "id",
        #     F.col("purchase_data.purchase_date").alias("purchase_time"),
        #     F.col("purchase_data.items").alias("frequency"),
        #     F.col("purchase_data.avg_price").alias("monetary")
        # ).collect()
        
        # for record in purchase_records:
        #     user_id = record["id"]
        #     if user_id not in analysis_results["rfm_data"]:
        #         analysis_results["rfm_data"][user_id] = {
        #             "last_purchase": record["purchase_time"],
        #             "frequency": record["frequency"],
        #             "monetary": record["monetary"]
        #         }
        #     else:
        #         analysis_results["rfm_data"][user_id]["frequency"] += record["frequency"]
        #         analysis_results["rfm_data"][user_id]["monetary"] += record["monetary"]
        #         if record["purchase_time"] > analysis_results["rfm_data"][user_id]["last_purchase"]:
        #             analysis_results["rfm_data"][user_id]["last_purchase"] = record["purchase_time"]
        
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
    data_dir = "/data2/wyh-dataset/30G_dataset/"
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
        # 4. 建立用户画像
        process_partition(df)    

    # 5. 打印数据质量报告
    print("\n数据质量报告:")
    print("\n缺失值统计:")
    for col, count in preprocess_analysis["missing_values"].items():
        print(f"列 {col}: 缺失值数量 = {count}, 缺失比例 = {preprocess_analysis['missing_percent'][col]:.2f}%")

    # 6.执行所有可视化函数
    plot_gender_distribution()
    plot_age_distribution()
    plot_login_time_distribution()
    plot_country_distribution()


    print(f"所有可视化图表已保存到 {output_dir}")
    end_time = time.time()
    print(f"处理完成，总耗时: {end_time - start_time:.2f}秒")
    spark.stop()