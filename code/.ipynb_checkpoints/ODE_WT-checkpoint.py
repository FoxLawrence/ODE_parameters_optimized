import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import smtplib
from email.mime.text import MIMEText
from email.header import Header
    
np.seterr(all='ignore')
np.seterr(over='ignore')
cpu_core = 1 # 更改使用的cpu核心数

# === Definition of path function ===
def get_paths(keyword):
    return {
        "data": f"../exp_data/{keyword}_data.csv",          
        "opt_par_csv": f"../opt_par/{keyword}/parameters_{keyword}.csv",
        "output_plot": f"../plot/{keyword}/compare_logx_{keyword}.png",
        "output_csv": f"../simulation_data/{keyword}/simulations_{keyword}.csv"
        
    }

# Which sample data used 
sample_name = "WT"
PATHS = get_paths(sample_name)

experimental_data = pd.read_csv(PATHS["data"])  # 
t_exp = experimental_data['time'].values #
FLY_exp = experimental_data['values'].values #

df_opt_par = pd.read_csv(PATHS["opt_par_csv"], header = 0, names=['key', 'value'])
k_1 = 4e6
sk = 1e-06
# 定义参数 p means parameters
p_opt = dict(zip(df_opt_par['key'], df_opt_par['value']))
print(p_opt)