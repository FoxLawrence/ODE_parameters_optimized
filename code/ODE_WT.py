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

p_raw = dict(
    k41 = 0.18,
    KK41 = 1000,
    k51 = 10000,
    k52 = 8,
    k53 = 500000,
    k54 = 6,
    
    
    kpq = 10000,


    pot1 = 0.0016,
    pot2 = 0.0046,
    pot3 = 0.003,
    kp680 = 1100000,
    tauc = 0.06,
    kcar = 10,
    kda = 150000,
    cm = 5,
    d = 0.5,
#    sk=1e-6, # 荧光比例常数，线性缩放FLY幅值 1e-16
    
    #tauw1 = 0.4 #未被使用

    tauhp = 10,
    tauhp1 = 100,
    tauhn = 720,
    taupt = 250,
    taupt1 = 500,
)


# 添加变量名列表S
S = ['x1','x2','x3','x4','x5','x6','x7',
     'y1','y2','y3','y4','y5','y6','y7',
     'z1','z2','z3','z4','z5','z6','z7',
     'PQH2',
     'g1','g2','g3','g4','g5','g6','g7',
     'PQ']    

# 添加索引
I = {s:i for i,s in enumerate(S)}

y0 = np.zeros(len(S))
y0[I['x1']] = 0.9
y0[I['g1']] = 0.62
y0[I['PQ']] = 15.4
y0[I['PQH2']] = 0.8


# 定义时间间隔 1µs … 10s 10** 对结果取10的幂 最后一个参数决定了取多少个间隔
t_eval = 10**np.linspace(-6, 4, 601)


# ======================
# 反应网络ODE定义
# ======================

def reaction_system(params):


    del1 = 0.5
    del2 = 0.3
    del3 = 0.1
    del4 = 0.1
    Pool1 = 16.2
    Pool2 = 1.62

    k1= 3e9  #原数值0.54 介于k3 = 3e9 fixed 
    # k_1= 4e6 # fixed from article
    # antn=100, # fixed article is 112
    k2 = 320000000 # fixed 32000000000/100
    KK2 = 20 # fixed from article
    k3 = 3e9 # fixed from article
    KK3 = 1e7 # fixed from article
    KK4 = 80 # fixed from article
    tauw = 0.1 # fixed from article
    #k4 = 5e4 * np.exp(-t/tauw)+3.8, # fixed from article 
    k7 = 2500 # fixed from article
    KK7 = 12.5 # fixed from article
    k14 = 1600 # fixed from article
    KK14 = 10 # fixed from article
    k21 = 800 # fixed from article and k21-27 = 800
    KK21 = 10 # fixed from article
    k34 = 100 # fixed from article and k34-40 = 100
    KK34 = 20 # fixed from article
   
    k41 = params['k41']
    KK41 = params['KK41']
    k51 = params['k51']
    k52 = params['k52']
    k53 = params['k53']
    k54 = params['k54']
    kpq = params['kpq']
    pot1 = params['pot1']
    pot2 = params['pot2']
    pot3 = params['pot3']
    kp680 = params['kp680']
    tauc = params['tauc']
    kcar = params['kcar']
    kda = params['kda']
    cm = params['cm']
    d = params['d']
    # sk= 1e-6 # 荧光比例常数，线性缩放FLY幅值 1e-16
    
    #tauw1 = 0.4 #未被使用

    tauhp = params['tauhp']
    tauhp1 = params['tauhp1']
    tauhn = params['tauhn']
    taupt = params['taupt']
    taupt1 = params['taupt1']
    
    k5 = k2/10
    k6 = k1
    k8 = k1
    k9 = k1
    k10 = k1
    #k11 = k4
    k13 = k6
    k22 = k21
    k23 = k21
    k24 = k21
    k25 = k21
    k26 = k21
    k27 = k21
    k35 = k34
    k36 = k34
    k37 = k34
    k38 = k34
    k39 = k34
    k40 = k34

    KK5 = KK2/8
    KK13 = KK5
    k_2 = k2/KK2
    k_3 = k3/KK3
    #k_4 = k4/KK4
    k_5 = k5/KK5
    k_7 = k7/KK7
    k_14 = k14/KK14
    k_21 = k21/KK21
    k_34 = k34/KK34
    k_41 = k41/KK41
    k_13 = k_5

    def rhs(t, y):
        (x1, x2, x3, x4, x5, x6, x7,
         y1, y2, y3, y4, y5, y6, y7,
         z1, z2, z3, z4, z5, z6, z7, 
         PQH2,
         g1, g2, g3, g4, g5, g6, g7,
         PQ) = y

        k4 = 5e4 * np.exp(-t/tauw)+3.8
        k11 = k4
        k_4 = k4/KK4
    # 动态参数计算
        Hp = 0.00024 + 0.00126*(1 - np.exp(-t/tauhp)) + 0.0004 * ( 1 - np.exp(-t/tauhp1))
        Hn = 0.00008 - 0.000025*(1 - np.exp(-t/tauhn))
        pt = pot1 - pot2 * np.exp(-t/taupt) + pot3 * np.exp(-t/taupt1)
        pt1 = d*pt*3760/cm
    
    
        kpot1 = np.exp(-del1*pt1)  
        k_pot1 = np.exp(del1*pt1)
        kpot2 = np.exp(-del2*pt1)
        k_pot2 = np.exp(del2*pt1)
        kpot3 = np.exp(-del3*pt1)
        k_pot3 = np.exp(del3*pt1)
        kpot4 = np.exp(-del4*pt1)
        k_pot4 = np.exp(del4*pt1)
        kz = Hn*10000
    
        PQ=-x1-x2-x3-x4-x5-x6-x7-y1-y2-y3-y4-y5-y6-y7-z1-z2-z3-z4-z5-z6-z7-PQH2+Pool1
        g7=-x1-x2-x3-x4-x5-x6-x7-y1-y2-y3-y4-y5-y6-y7-z1-z2-z3-z4-z5-z6-z7-g1-g2-g3-g4-g5-g6+Pool2
    
    
        
        # 中间变量
        p680 = x3 + x4 + g3 + g4 + y3 + y4 + z3 + z4 + x7 + g7 + y7 + z7
        car = kcar * np.exp(-t/tauc) 
        
        # 反应速率计算 (V[1]-V[53] 精选关键反应)
        V1 = k1*x1 - (k_1 + kp680 * p680 + car)* x2  
        V2 = k2*x2*kpot1 - k_2 * x3 * k_pot1
        V3 = k3*x3*kpot2 - k_3 * x4 * k_pot2
        V4 = k4*x4*kpot3 - k_4 * x5 * k_pot3 * Hp * 10000
        V5 = k1 * x5 - (k_1 + kp680 * p680 + kda +car + kpq * PQ) * x6
        V6 = k5 * x6 * kpot1 - k_5 * x7 * k_pot1
        V7 = k7 * x5 - k_7 * y1
        V8 = k8 * y1 - (k_1 + kp680 * p680 + car) * y2
        V9 = k2 * y2 * kpot1 - k_2 * y3 * k_pot1
        V10 = k3 * y3 * kpot2 - k_3 * y4 * k_pot2
        V11 = k4 * y4 * kpot3 - k_4 * y5 * k_pot3 * Hp * 10000
        V12 = k9 * y5 - (k_1 + kp680 * p680 + kda + car + kpq * PQ) * y6
        V13 = k5 * y6 * kpot1 - k_13 * y7 * k_pot1
        V14 = k14 * y5 * kz * kpot4 - k_14 * z1 * k_pot4
        V15 = k10 * z1 - (k_1 + kp680 * p680 + kda + car + kpq * PQ) * z2
        V16 = k2 * z2 * kpot1 - k_2 * z3 * k_pot1
        V17 = k3 * z3 * kpot2 - k_3 * z4 * k_pot2
        V18 = k4 * z4 * kpot3 - k_4 * z5 * k_pot3 * Hp * 10000
        V19 = k11 * z5 - (k_1 + kp680 * p680 + kda + car + kpq*PQ)*z6
        V20 = k5 * z6 * kpot1 - k_13 * z7 * k_pot1
        V21 = k21 * z1 * kz * kpot4 - k_21 * g1 * PQH2 * k_pot4 # adjust PQH2 and  
        V22 = k22 * z2 * kz * kpot4 - k_21 * g2 * PQH2 * k_pot4 # adjust PQH2 and 
        V23 = k23 * z3 * kz * kpot4 - k_21 * g3 * PQH2 * k_pot4 # adjust PQH2 and 
        V24 = k24 * z4 * kz * kpot4 - k_21 * g4 * PQH2 * k_pot4 # adjust PQH2 and 
        V25 = k25 * z5 * kz * kpot4 - k_21 * g5 * PQH2 * k_pot4 # adjust PQH2 and 
        V26 = k26 * z6 * kz * kpot4 - k_21 * g6 * PQH2 * k_pot4 # adjust PQH2 and 
        V27 = k27 * z7 * kz * kpot4 - k_21 * g7 * PQH2 * k_pot4 # adjust PQH2 and 
        V28 = k1 * g1 - (k_1 + kp680 * p680 + car) * g2
        V29 = k2 * g2 * kpot1 - k_2 * g3 * k_pot1
        V30 = k3 * g3 * kpot2 - k_3 * g4 * k_pot2
        V31 = k4 * g4 * kpot3 - k_4 * g5 * k_pot3 * Hp * 10000
        V32 = k6 * g5 - (k_1 + kp680 * p680 + kda + car +kpq * PQ) * g6
        V33 = k5 * g6 * kpot1 - k_5 * g7 * k_pot1
        V34 = k34 * g1 * PQ - k_34 * x1 # PQ in
        V35 = k35 * g2 * PQ - k_34 * x2 # PQ in
        V36 = k36 * g3 * PQ - k_34 * x3 # PQ in
        V37 = k37 * g4 * PQ - k_34 * x4 # PQ in
        V38 = k38 * g5 * PQ - k_34 * x5 # PQ in
        V39 = k39 * g6 * PQ - k_34 * x6 # PQ in
        V40 = k40 * g7 * PQ - k_34 * x7 # PQ in
        V41 = k41 * PQH2 * kpot3 - k_41 * PQ * k_pot3 * np.power((Hp*10000),2) # PQ out
        V42 = k51 * x3
        V43 = k54 * x4
        V44 = k53 * x7
        V45 = k51 * y3
        V46 = k52 * y4
        V47 = k53 * y7
        V48 = k51 * z3
        V49 = k52 * z4
        V50 = k53 * z7
        V51 = k51 * g3 
        V52 = k54 * g4
        V53 = k53 * g7
        vwoc = V11 + V4 + V18 + V31
        
       
        # 原始状态变量
        PQ = -V34 - V35 - V36 - V37 - V38 - V39 - V40 + V41
        PQH2 = V21 + V22 + V23 + V24 + V25 + V26 + V27 - V41
    
    
        dydt = np.zeros_like(y)
        dydt[I['x1']] = -V1 + V34 + V42 + V43  # for x1
        dydt[I['x2']] = V1 - V2 + V35          # for x2
        dydt[I['x3']] = V2 - V3 + V36 - V42    # for x3
        dydt[I['x4']] = V3 - V4 + V37 - V43    # for x4
        dydt[I['x5']] = V4 - V5 - V7 + V38 + V44  # for x5
        dydt[I['x6']] = V5 - V6 + V39          # for x6
        dydt[I['x7']] = V6 + V40 - V44         # for x7
        dydt[I['y1']] = V7 - V8 + V45 + V46    # for y1
        dydt[I['y2']] = V8 - V9                # for y2
        dydt[I['y3']] = V9 - V10 - V45         # for y3
        dydt[I['y4']] = V10 - V11 - V46       # for y4
        dydt[I['y5']] = V11 - V12 - V14 + V47 # for y5
        dydt[I['y6']] = V12 - V13             # for y6
        dydt[I['y7']] = V13 - V47             # for y7
        dydt[I['z1']] = V14 - V15 - V21 + V48 + V49  # for z1
        dydt[I['z2']] = V15 - V16 - V22       # for z2
        dydt[I['z3']] = V16 - V17 - V23 - V48 # for z3
        dydt[I['z4']] = V17 - V18 - V24 - V49 # for z4
        dydt[I['z5']] = V18 - V19 - V25 + V50 # for z5
        dydt[I['z6']] = V19 - V20 - V26       # for z6
        dydt[I['z7']] = V20 - V27 - V50       # for z7
        dydt[I['PQH2']] = V21 + V22 + V23 + V24 + V25 + V26 + V27 - V41  # for PQH2
        dydt[I['g1']] = V21 - V28 - V34 + V51 + V52  # for g1
        dydt[I['g2']] = V22 + V28 - V29 - V35  # for g2
        dydt[I['g3']] = V23 + V29 - V30 - V36 - V51  # for g3
        dydt[I['g4']] = V24 + V30 - V31 - V37 - V52  # for g4
        dydt[I['g5']] = V25 + V31 - V32 - V38 + V53  # for g5
        dydt[I['g6']] = V26 + V32 - V33 - V39  # for g6
        dydt[I['g7']] = V27 + V33 - V40 - V53  # for g7
        dydt[I['PQ']] = -V34 - V35 - V36 - V37 - V38 - V39 - V40 + V41  # for PQ
    
        # FLY的计算由 simluate_system函数中的obs计算，使用obs['FLY']调用
        # FLY = sk * k_1 * (x2 + x6 + y2 + y6 + g2 + g6 + z2 + z6)
        # p680_obs = x3 + x4 + g3 + g4 + y3 + y4 + z3 + z4 + x7 + g7 + y7 + z7
        # pott = 2000000 * pt
    
        # Calculate pH values
        # pHn = -np.log10(0.001 * Hn) * 1000  
        # pHp = -np.log10(0.001 * Hp) * 1000 
        # dpH = 1 * (pHn - pHp)
    
        # Calculate xz values
        # xz2 = 18000 * np.power( 10, 6) * (x2 + y2 + g2 + z2) / 1.62
        # xz6 = 18000 * np.power( 10, 6) * (x6 + y6 + g6 + z6) / 1.62
        # xz4 = 8000 * ( x4 + g4 + y4 + z4 ) / 1.62
        # xz5 = 8000 * ( x5 + g5 + y5 + z5) / 1.62
        # xz45 = xz4 + xz5 
        # yy1 = 10000 * y1 / 1.62
        # yy17 = 1000 * ( y1 + y4 + y5 + y7 + y3 + y2 + y6) / 1.62
        
        # xz1 = 4 * (x1 + g1 + y1 + z1) / 1.62
        # xz347 = 40 * p680 / 1.62
        # Calculate pqhh
        # pqhh = 10000 * PQH2 / ( PQ + PQH2)
    
        # # Calculate remaining values
        # xg4 = ( x4 + g4 ) * 40 /1.62
        # xx5 = x5 * 7050 / 1.62
        # yy5 = y5 * 7050 / 1.62
        # zz5 = z5 * 7050 / 1.62
        # gg5 = g5 * 7050 / 1.62
        # xg5 = xx5 + gg5
        # vv41 = 10 * V41
        
        return dydt

    return rhs



# 对反应方程进行ODE求解
def simulate_system():
    results = {
        'time': [],
        'FLY': []
    }
    # solver = reaction_system(t,y)
    solver_opt = reaction_system(p_opt)
    # 可能需要添加第一步的时间要到e-10的量级
    sol_opt = solve_ivp(
        solver_opt,
        (t_eval[0], t_eval[-1]), # 设置步长范围
        y0,                      # 设置ODE方程
        t_eval = t_exp,
        method = 'LSODA',          # 设置计算方法 ‘BDF’ 'LSODA'
        rtol = 1e-5,
        atol = 1e-7,
        first_step = 1e-10
    )



    Y_opt = sol_opt.y.T

    
    obs_opt = dict(
        FLY = sk* k_1*(Y_opt[:,I['x2']]+Y_opt[:,I['x6']]
                                +Y_opt[:,I['y2']]+Y_opt[:,I['y6']]
                                +Y_opt[:,I['z2']]+Y_opt[:,I['z6']]
                                +Y_opt[:,I['g2']]+Y_opt[:,I['g6']]),
    )
    df_opt = pd.DataFrame({
        'time': sol_opt.t,
        'FLY': obs_opt['FLY']
    }).dropna()

    
    solver_rhs = reaction_system(p_raw)
    sol_raw = solve_ivp(
        solver_rhs,
        (t_eval[0], t_eval[-1]), # 设置步长范围
        y0,                      # 设置ODE方程
        t_eval = t_exp,
        method = 'LSODA',          # 设置计算方法 ‘BDF’ 'LSODA'
        rtol = 1e-5,
        atol = 1e-7,
        first_step = 1e-10
    )



    Y = sol_raw.y.T

    obs_raw = dict(
        FLY = sk* k_1*(Y[:,I['x2']]+Y[:,I['x6']]
                                +Y[:,I['y2']]+Y[:,I['y6']]
                                +Y[:,I['z2']]+Y[:,I['z6']]
                                +Y[:,I['g2']]+Y[:,I['g6']]),
    )
    
    # 作图
    time_log = np.log10(sol_opt.t*1e3) # 以毫秒作单位
    plt.figure(figsize=(12, 6))
    
    # 绘制实验数据（散点）和模型数据（曲线）
    plt.scatter(t_exp, FLY_exp, s=10, label='Experiment Data')
    plt.plot(t_exp, obs_raw['FLY'], '-', label='Model_raw', linewidth=2)
    plt.plot(t_exp, obs_opt['FLY'], '-', label='Model_opt', linewidth=2)
    
    # 设置 x 轴为对数坐标
    plt.xscale('log')  # 以 10 为底的对数坐标
    
    # 设置坐标轴标签、标题和图例
    plt.xlabel('Time (ms) (log scale)')  # 添加 log scale 说明
    plt.ylabel('FLY Intensity')
    plt.title('Model Data (Logarithmic Time Axis)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # 同时显示主次网格线
    
    # 保存图像并显示
    plt.savefig(PATHS["output_plot"], dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_opt
results_df = simulate_system()

# 保存数据到CSV
results_df.to_csv(PATHS["output_csv"], index=False)
print("Simulation completed. Results saved to simulation_results.csv")
