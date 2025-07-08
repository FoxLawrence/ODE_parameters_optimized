import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# 配置参数
INPUT_FILE = "rawdata.txt"    # 输入数据文件
OUTPUT_NAMES = {             # 输出文件基础名配置
    1: "WT",    # y1列对应的输出文件名
    2: "dPBS",  # y2列对应的输出文件名
    3: "dOCP"   # y3列对应的输出文件名
}

def read_tab_separated_file(filename):
    """读取TAB分隔的TXT文件，返回所有列数据"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 4:  # 确保有4列数据
                try:
                    # 读取所有4列数据
                    row = [float(parts[0])] + [float(p) if p.strip() else np.nan for p in parts[1:4]]
                    data.append(row)
                except ValueError:
                    continue
    return np.array(data)

def hybrid_processing(t_exp, FLY_exp, density_thresh=0.03, base_spacing=0.05):
    """智能分段处理数据"""
    log_t = np.log10(t_exp)
    log_diff = np.diff(log_t)
    
    # 密集区处理
    is_dense = np.insert(log_diff < density_thresh, 0, False)
    keep_mask = np.ones_like(t_exp, dtype=bool)
    
    if len(np.where(is_dense)[0]) > 0:
        for cluster in np.split(np.where(is_dense)[0], np.where(np.diff(np.where(is_dense)[0]) > 1)[0]+1):
            if len(cluster) > 2:
                sub_log_t = log_t[cluster]
                keep_sub = [cluster[0]]
                last_kept = sub_log_t[0]
                for i in range(1, len(cluster)-1):
                    if sub_log_t[i] - last_kept >= base_spacing:
                        keep_sub.append(cluster[i])
                        last_kept = sub_log_t[i]
                keep_sub.append(cluster[-1])
                keep_mask[cluster] = False
                keep_mask[keep_sub] = True
    
    # 稀疏区插值
    t_densered, fly_densered = t_exp[keep_mask], FLY_exp[keep_mask]
    log_t_sparse = np.log10(t_densered)
    interp_func = interp1d(log_t_sparse, fly_densered, kind='linear', fill_value="extrapolate")
    
    final_log_t = []
    for i in range(len(log_t_sparse)-1):
        final_log_t.append(log_t_sparse[i])
        spacing = log_t_sparse[i+1] - log_t_sparse[i]
        if spacing > density_thresh:
            n_insert = max(1, int(spacing / base_spacing * 0.8))
            new_points = np.linspace(log_t_sparse[i], log_t_sparse[i+1], n_insert + 2)[1:-1]
            final_log_t.extend(new_points)
    
    final_log_t.append(log_t_sparse[-1])
    return 10**np.array(final_log_t), interp_func(np.array(final_log_t))

def save_results(base_name, t_processed, fly_processed, t_original=None, fly_original=None):
    """保存结果到指定文件"""
    os.makedirs("OUTPUT", exist_ok=True)
    
    # 保存CSV数据
    np.savetxt(f"OUTPUT/{base_name}.csv", 
               np.column_stack((t_processed, fly_processed)),
               delimiter=',', fmt='%.6f', header='x,y', comments='')
    
    # 保存对比图
    if t_original is not None and fly_original is not None:
        plt.figure(figsize=(10, 6))
        plt.xscale('log')
        plt.plot(t_original, fly_original, 'b-', alpha=0.5, label='Original')
        plt.plot(t_processed, fly_processed, 'r-', label='Processed')
        plt.title(base_name)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"OUTPUT/{base_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 读取数据
    try:
        data = read_tab_separated_file(INPUT_FILE)
        if len(data) == 0:
            raise ValueError("No valid data found in the file")
        t_exp = data[:, 0]
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # 处理每一列数据
    for col in [1, 2, 3]:  # 处理y1, y2, y3列
        try:
            # 提取当前列数据，过滤掉NaN值
            valid_mask = ~np.isnan(data[:, col])
            t_valid = t_exp[valid_mask]
            y_valid = data[valid_mask, col]
            
            # 数据处理
            t_processed, y_processed = hybrid_processing(t_valid, y_valid)
            
            # 保存结果
            save_results(OUTPUT_NAMES[col], t_processed, y_processed, t_valid, y_valid)
            print(f"Processed {OUTPUT_NAMES[col]} data saved to OUTPUT/{OUTPUT_NAMES[col]}.csv/.png")
        except Exception as e:
            print(f"Error processing column {col}: {e}")

if __name__ == "__main__":
    main()