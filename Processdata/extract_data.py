import pandas as pd

# 读取原始数据（假设是制表符分隔的文本文件）
# 如果格式不同，请调整sep参数，如sep=','表示逗号分隔
raw_data = pd.read_csv('rawdata.txt', sep='\t', header=None, 
                      names=['time', 'y1', 'y2', 'y3'])

# 生成y1数据文件（自动处理空值）
y1_data = raw_data[['time', 'y1']].dropna(subset=['y1'])
y1_data.to_csv('y1_data.csv', index=False, header=['time', 'values'])

# 生成y2数据文件（自动处理空值）
y2_data = raw_data[['time', 'y2']].dropna(subset=['y2'])
y2_data.to_csv('y2_data.csv', index=False, header=['time', 'values'])

# 生成y3数据文件（假设y3没有空值）
y3_data = raw_data[['time', 'y3']]
y3_data.to_csv('y3_data.csv', index=False, header=['time', 'values'])

print("3个CSV文件已成功生成：")
print("- y1_data.csv")
print("- y2_data.csv")
print("- y3_data.csv")