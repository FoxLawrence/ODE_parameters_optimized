#
//  Douglas-Peuker.py
//  Processdata
//
//  Created by 王中杰 on 2025/6/12.
//
import numpy as np
import time
import matplotlib.pyplot as plt

def read_tab_separated_file(filename):
    """
    读取TAB分隔的TXT文件，返回x和多个y列的数组
    文件格式：第一列为x，其他列为y1, y2, y3...
    """
    data = np.loadtxt(filename, delimiter='\t')
    x = data[:, 0]
    y_columns = data[:, 1:]
    return x, y_columns

def perpendicular_distance(point, line_start, line_end):
    """
    计算点到线段的垂直距离
    """
    # 线段长度为0的情况
    if np.allclose(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    # 计算线段向量和点向量
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    # 线段长度的平方
    line_length_sq = np.sum(line_vec**2)
    
    # 计算投影比例
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_sq))
    
    # 计算投影点
    projection = line_start + t * line_vec
    
    # 返回点到投影点的距离
    return np.linalg.norm(point - projection)

def douglas_peucker(points, epsilon, start_idx, end_idx, indices=None):
    """
    Douglas-Peucker算法的递归实现
    """
    if indices is None:
        indices = set()
    
    # 添加起点和终点
    indices.add(start_idx)
    indices.add(end_idx)
    
    # 如果起点和终点相同或相邻，直接返回
    if end_idx <= start_idx + 1:
        return indices
    
    # 找到离线段最远的点
    max_distance = 0.0
    max_index = start_idx
    
    line_start = points[start_idx]
    line_end = points[end_idx]
    
    for i in range(start_idx + 1, end_idx):
        distance = perpendicular_distance(points[i], line_start, line_end)
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # 如果最大距离大于epsilon，则递归处理
    if max_distance > epsilon:
        indices = douglas_peucker(points, epsilon, start_idx, max_index, indices)
        indices = douglas_peucker(points, epsilon, max_index, end_idx, indices)
    
    return indices

def optimized_douglas_peucker(points, epsilon):
    """
    优化的Douglas-Peucker算法实现
    使用迭代代替递归，避免栈溢出和提高性能
    """
    if len(points) < 3:
        return points
    
    # 使用栈来模拟递归
    stack = []
    keep = set()
    
    # 初始范围是整个数组
    stack.append((0, len(points) - 1))
    
    while stack:
        start_idx, end_idx = stack.pop()
        
        # 添加当前范围的起点和终点
        keep.add(start_idx)
        keep.add(end_idx)
        
        if end_idx <= start_idx + 1:
            continue
        
        # 找到离线段最远的点
        max_distance = 0.0
        max_index = start_idx
        
        line_start = points[start_idx]
        line_end = points[end_idx]
        
        for i in range(start_idx + 1, end_idx):
            distance = perpendicular_distance(points[i], line_start, line_end)
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # 如果最大距离大于epsilon，则将子范围压入栈
        if max_distance > epsilon:
            stack.append((start_idx, max_index))
            stack.append((max_index, end_idx))
    
    # 将保留的索引排序并提取对应的点
    sorted_indices = sorted(keep)
    return points[sorted_indices]

def process_all_curves(x, y_columns, epsilon):
    """
    处理所有y列的曲线
    """
    simplified_curves = []
    processing_times = []
    
    for i in range(y_columns.shape[1]):
        y = y_columns[:, i]
        points = np.column_stack((x, y))
        
        # 计时
        start_time = time.time()
        
        # 使用优化的Douglas-Peucker算法
        simplified_points = optimized_douglas_peucker(points, epsilon)
        
        # 记录处理时间
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        simplified_curves.append(simplified_points)
    
    return simplified_curves, processing_times

def plot_results(original_x, original_y, simplified_curves, y_labels=None):
    """
    绘制原始曲线和简化后的曲线
    """
    if y_labels is None:
        y_labels = [f"Y{i+1}" for i in range(len(simplified_curves))]
    
    plt.figure(figsize=(12, 8))
    
    for i, (orig_y, simp_points) in enumerate(zip(original_y.T, simplified_curves)):
        plt.subplot(len(simplified_curves), 1, i+1)
        plt.plot(original_x, orig_y, 'b-', label='Original', alpha=0.5)
        plt.plot(simp_points[:, 0], simp_points[:, 1], 'r-', label='Simplified')
        plt.title(y_labels[i] if i < len(y_labels) else f"Curve {i+1}")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # 文件路径
    filename = 'rawdata.txt'  # 替换为你的文件路径
    
    # 读取数据
    try:
        x, y_columns = read_tab_separated_file(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # DP算法的epsilon值（可调整）
    epsilon = 0.00001 #给kk 的DBsolve用0.001
    
    
    # 处理所有曲线
    simplified_curves, processing_times = process_all_curves(x, y_columns, epsilon)
    
    # 打印处理时间
    #for i, time_taken in enumerate(processing_times):
     #   print(f"Curve {i+1} processing time: {time_taken:.4f} seconds")
    
    # 绘制结果
    #plot_results(x, y_columns, simplified_curves)
    
    # 保存结果到文件
    for i, curve in enumerate(simplified_curves):
        output_filename = f'simplified_curve_{i+1}.txt'
        np.savetxt(output_filename, curve, delimiter='\t', fmt='%.6f')
        print(f"Simplified curve {i+1} saved to {output_filename}")

if __name__ == "__main__":
    main()
