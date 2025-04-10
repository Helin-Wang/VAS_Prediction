import pandas as pd
import re

# 读取 Excel 文件
file_path = 'dataset.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 定义处理函数
def process_imaging(exam):
    if re.match(r'左|右', exam):  # 匹配左膝或右膝
        return 1
    elif re.match(r'双', exam):  # 匹配双膝
        return 2
    else:
        return 0  # 无关数据返回 0

# 应用函数处理列
data['影像学检查'] = data['影像学检查'].apply(process_imaging)
print(data['影像学检查'].value_counts())

output_file = 'dataset.xlsx'  # 文件名
data.to_excel(output_file, index=False)  # 不保存索引，指定表名

