import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

output_dir = "/home/wangyuhui/data mining"

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 14

def plot_gender_distribution():
    gender_dist = {
        '男': 1200,
        '女': 800
    }
    
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
        plt.yticks(range(0, max(sizes) + 1, 200))
        
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
    age_dist = {
        '18-24岁': 300,
        '25-34岁': 500,
        '35-44岁': 400,
        '45-54岁': 200,
        '55岁及以上': 100
    }
    
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

# 国家分布（柱状图）
def plot_country_distribution():
    country_dist = {'德国':300, '美国': 500, '中国': 400, '法国': 200, '日本': 100, '英国': 150, '印度': 250, '巴西': 300, '俄罗斯': 200, '意大利': 100}
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
plot_country_distribution()

# def plot_login_time_distribution():
#     login_dist_by_age = {
#         "18-24岁": {"0-6clock": 100, "6-12clock": 300, "12-18clock": 500, "18-24clock": 300},
#         "25-34岁": {"0-6clock": 50, "6-12clock": 200, "12-18clock": 600, "18-24clock": 400},
#         "35-44岁": {"0-6clock": 30, "6-12clock": 150, "12-18clock": 400, "18-24clock": 350},
#         "45-54岁": {"0-6clock": 20, "6-12clock": 100, "12-18clock": 300, "18-24clock": 250},
#         "55岁及以上": {"0-6clock": 10, "6-12clock": 50, "12-18clock": 100, "18-24clock": 150}
#     }
#     # 提取数据
#     age_groups = list(login_dist_by_age.keys())
#     time_periods = ["0-6clock", "6-12clock", "12-18clock", "18-24clock"]
    
#     # 创建数据矩阵
#     data = []
#     for age in age_groups:
#         data.append([login_dist_by_age[age][period] for period in time_periods])
    
#     plt.figure(figsize=(12, 8))
    
#     # 绘制热力图
#     cmap = sns.cubehelix_palette(start=0, light=1, dark=0, as_cmap=True)  # 更美观的颜色映射
#     sns.heatmap(data, annot=True, fmt='d', xticklabels=time_periods, yticklabels=age_groups, cmap=cmap, linewidths=0.5, linecolor='white')
    
#     # 设置标题和标签字体
#     plt.title('不同年龄段用户登录时间分布热力图', fontsize=18)
#     plt.xlabel('时间段', fontsize=14)
#     plt.ylabel('年龄段', fontsize=14)
    
#     # 设置刻度和标签字体
#     plt.xticks(fontsize=12, rotation=45)
#     plt.yticks(fontsize=12)
    
#     # 添加网格和边框
#     plt.grid(True, linestyle='--', linewidth=0.5, color='white')
#     plt.gca().patch.set_edgecolor('black')
#     plt.gca().patch.set_linewidth(1)
    
#     # 调整布局
#     plt.tight_layout()
    
#     output_path = os.path.join(output_dir, "login_time_by_age.png")
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.close()

import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

output_dir = "/home/wangyuhui/data mining"

# 登录时间分布与年龄分布的二维数据
login_dist_by_age = {
    "18-24岁": {"0-6clock": 100, "6-12clock": 300, "12-18clock": 500, "18-24clock": 300},
    "25-34岁": {"0-6clock": 50, "6-12clock": 200, "12-18clock": 600, "18-24clock": 400},
    "35-44岁": {"0-6clock": 30, "6-12clock": 150, "12-18clock": 400, "18-24clock": 350},
    "45-54岁": {"0-6clock": 20, "6-12clock": 100, "12-18clock": 300, "18-24clock": 250},
    "55岁及以上": {"0-6clock": 10, "6-12clock": 50, "12-18clock": 100, "18-24clock": 150}
}

def plot_login_time_by_age():
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

plot_login_time_by_age()
plot_age_distribution()
plot_gender_distribution()
# plot_login_time_distribution()