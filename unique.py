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

# 初始化Spark会话
spark = SparkSession.builder \
    .appName("UserBehaviorAnalysis") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# 定义数据结构和全局变量
# ... 之前的定义保持不变 ...

# 检查id是否唯一
def check_unique_id(df):
    total_count = df.count()
    unique_count = df.select(F.countDistinct("id")).collect()[0][0]
    
    if total_count == unique_count:
        print("id字段是唯一的。")
    else:
        print(f"id字段不是唯一的。总记录数：{total_count}，唯一id数：{unique_count}")

if __name__ == "__main__":
    # 1. 获取所有分区文件并加载整个数据集
    data_dir = "/data2/wyh-dataset/10G_dataset/"
    # 加载整个数据集
    df = spark.read.parquet(data_dir)
    
    # 记录开始时间
    start_time = time.time()

    # 检查id是否唯一
    check_unique_id(df)

    end_time = time.time()
    print(f"处理完成，总耗时: {end_time - start_time:.2f}秒")
    spark.stop()