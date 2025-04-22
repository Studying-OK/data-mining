from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, ArrayType, TimestampType
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from matplotlib import cm
import numpy as np
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
# 设置可视化风格
# --------------------------------
# 设置可视化风格
plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
analyse_dir = "/home/wangyuhui/data mining"
output_dir = os.path.join(analyse_dir, "visualization")
os.makedirs(output_dir, exist_ok=True)

# 性别分布（柱状图）
def plot_gender_distribution():
    gender_dist = analysis_results["user_profile"]["gender_dist"]
    
    if gender_dist:
        labels = list(gender_dist.keys())
        sizes = list(gender_dist.values())
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=sizes, palette='viridis', width=0.3)
        
        # 设置标题和坐标轴标签
        plt.title('用户性别分布')
        plt.xlabel('性别')
        plt.ylabel('用户数量')
        
        # 设置坐标轴范围和刻度
        plt.ylim(0, max(sizes) * 1.1)
        if max(sizes) > 50000000:
            plt.yticks(range(0, max(sizes) + 1, 10000000))
        elif max(sizes) > 20000000:
            plt.yticks(range(0, max(sizes) + 1, 5000000))
        else:
            plt.yticks(range(0, max(sizes) + 1, 100000))
        
        # 添加数据标签
        for index, value in enumerate(sizes):
            plt.text(index, value + 10, str(value), ha='center', va='bottom',)
        
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
            
            plt.figure(figsize=(10, 7))
            # 使用更丰富的颜色
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            plt.title('用户年龄分布', fontsize=18)
            
            # 添加图例
            # plt.legend(labels, loc='upper right', fontsize=12)
            plt.legend(bbox_to_anchor=(1.00, 1.05), fontsize=16)
            
            output_path = os.path.join(output_dir, "age_distribution.png")
            plt.savefig(output_path)
            plt.close()
    else:
        print("年龄分布数据为空，跳过可视化。")

# 登录时间分布（热力图）
def plot_login_time_distribution():
    login_dist_by_age = analysis_results["user_profile"]["always_login_dist"]
    
    # 提取数据
    age_groups = list(login_dist_by_age.keys())
    time_periods = ["0-6clock", "6-12clock", "12-18clock", "18-24clock"]
    
    # 创建数据矩阵
    data = []
    for age in age_groups:
        data.append([login_dist_by_age[age][period] for period in time_periods])
    
    plt.figure(figsize=(12, 8))
    
    # 绘制热力图
    cmap = sns.cubehelix_palette(start=0, light=1, dark=0, as_cmap=True)  
    sns.heatmap(data, annot=True, fmt='d', xticklabels=time_periods, yticklabels=age_groups, cmap=cmap, linewidths=0.5, linecolor='white')
    
    # 设置标题和标签字体
    plt.title('不同年龄段用户登录时间分布热力图', fontsize=18)
    plt.xlabel('时间段', fontsize=14)
    plt.ylabel('年龄段', fontsize=14)
    
    # 设置刻度和标签字体
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    
    # 添加网格和边框
    plt.grid(True, linestyle='--', linewidth=0.5, color='white')
    plt.gca().patch.set_edgecolor('black')
    plt.gca().patch.set_linewidth(1)
    
    # 调整布局
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "login_time_by_age.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# 登录时间分布（3D）
def plot_login_time_by_age():
    login_dist_by_age = analysis_results["user_profile"]["always_login_dist"]

        # 提取数据
    age_groups = list(login_dist_by_age.keys())
    time_periods = ["0-6clock", "6-12clock", "12-18clock", "18-24clock"]
    
    # 创建数据矩阵
    data = np.zeros((len(age_groups), len(time_periods)))
    for i, age in enumerate(age_groups):
        for j, period in enumerate(time_periods):
            data[i, j] = login_dist_by_age[age][period]
    
    # 创建3D热力图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建网格
    x = np.arange(len(time_periods))
    y = np.arange(len(age_groups))
    x, y = np.meshgrid(x, y)
    
    # 绘制3D热力图
    surf = ax.plot_surface(x, y, data, cmap=cm.inferno, linewidth=0, antialiased=False)
    
    # 设置标签和标题
    ax.set_xticks(np.arange(len(time_periods)))
    ax.set_xticklabels(time_periods, fontsize=12, rotation=45)
    ax.set_yticks(np.arange(len(age_groups)))
    ax.set_yticklabels(age_groups, fontsize=12)
    ax.set_zlabel('登录次数', fontsize=14)
    ax.set_xlabel('时间段', fontsize=14)
    ax.set_ylabel('年龄段', fontsize=14)
    ax.set_title('不同年龄段用户登录时间分布', fontsize=16)
    
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "login_time_by_age_3d.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# 国家分布（柱状图）
def plot_country_distribution():
    country_dist = analysis_results["user_profile"]["country_dist"]
    if country_dist:
        labels = list(country_dist.keys())
        sizes = list(country_dist.values())
        total_users = sum(sizes)
        normalized_sizes = [(count / total_users) * 100 for count in sizes]  # 归一化为百分比
        
        plt.figure(figsize=(12, 7))
        sns.barplot(x=labels, y=normalized_sizes, palette='husl', width=0.6)
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

        for age_group in analysis_results["user_profile"]["age_dist"].keys():
            # 根据年龄段筛选数据
            if age_group == "<18":
                age_filter = F.col("age") < 18
            elif age_group == "18-25":
                age_filter = (F.col("age") >= 18) & (F.col("age") < 25)
            elif age_group == "25-35":
                age_filter = (F.col("age") >= 25) & (F.col("age") < 35)
            elif age_group == "35-50":
                age_filter = (F.col("age") >= 35) & (F.col("age") < 50)
            elif age_group == ">50":
                age_filter = F.col("age") >= 50
            else:
                continue  # 跳过未知的年龄段
            
            df_age = df_exploded.filter(age_filter)
            
            for period, (start, end) in time_period_mapping.items():
                count = df_age.filter((F.col("hour") >= start) & (F.col("hour") < end)).count()
                analysis_results["user_profile"]["always_login_dist"][age_group][period] += count
    

        
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
    plot_login_time_by_age()
    plot_country_distribution()


    # print(f"总活跃用户数: {len(analysis_results['user_behavior']['active_users'])}")
    # print(f"总活跃用户占比: {len(analysis_results['user_behavior']['active_users'])/analysis_results['user_profile']['total_users']:.2%}")
    # print(f"总消费金额: {analysis_results['user_behavior']['total_spent']:.2f}")
    # print(f"平均每活跃用户消费金额: {analysis_results['user_behavior']['total_spent']/analysis_results['user_profile']['total_users']:.2%}")
    
    print(f"所有可视化图表已保存到 {output_dir}")
    end_time = time.time()
    print(f"处理完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 计算关键指标
    total_active_users = len(analysis_results['user_behavior']['active_users'])
    total_users = analysis_results['user_profile']['total_users']
    total_spent = analysis_results['user_behavior']['total_spent']

    # 构建要存储的 JSON 数据
    summary_results = {
        "total_active_users": total_active_users,
        "total_active_users_percentage": total_active_users / total_users,
        "total_spent": total_spent,
        "average_spent_per_active_user": total_spent / total_users,
        "visualization_output_dir": output_dir,
        "processing_time": end_time - start_time
    }

    # 将 summary_results 转换为 JSON 格式
    summary_json = json.dumps(summary_results, indent=4)

    # 定义输出文件路径
    summary_output_file = os.path.join(output_dir, "summary_results.json")

    # 将 JSON 数据写入文件
    with open(summary_output_file, 'w') as f:
        f.write(summary_json)

    print(f"关键分析指标已保存到 {summary_output_file}")
    spark.stop()