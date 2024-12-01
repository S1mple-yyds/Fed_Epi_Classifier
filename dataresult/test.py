import pandas as pd

# 读取CSV文件
df = pd.read_csv('df_list_1.csv')

# 计算行数
rows_to_select = int(10052 * 0.09)

# 获取第一列的前rows_to_select行
first_column_data = df.iloc[:rows_to_select, 5]

# 查看数据分布
data_distribution = first_column_data.value_counts()

# 如果需要，可以对结果进行排序
# data_distribution = data_distribution.sort_index() # 按索引排序
# data_distribution = data_distribution.sort_values() # 按值排序

print(data_distribution)