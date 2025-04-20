
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType,ArrayType
import os

# 初始化Spark会话
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .getOrCreate()

# 定义数据集路径
data_dir = "/data2/wyh-dataset/10G_dataset"
partition_path = os.path.join(data_dir, "part-00000.parquet")  # 示例分区文件

# 加载单个分区数据并打印Schema
df = spark.read.parquet(partition_path)
print("分区Schema:")
df.printSchema()

# 查看原始数据中 purchase_history 列的内容
print("\n原始数据中 purchase_history 列的内容示例：")
df.select("last_login").show(5, truncate=False)
df.select("income").show(5, truncate=False)
df.select("login_history").show(5, truncate=False)


# 定义购买记录的JSON结构
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

# 解析购买记录字段并打印示例
df_parsed = df.withColumn(
    "purchase_data",
    F.from_json(F.col("purchase_history"), purchase_schema)
)

print("\n解析后的购买记录列示例:")
df_parsed.select("purchase_data.*").show(5, truncate=False)