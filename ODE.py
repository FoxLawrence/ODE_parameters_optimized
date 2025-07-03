# 仅包含ODE部分，注意 k_1 
# 对反应方程进行ODE求解
@timeit
def simulate_system():
    results = {
        'time': [],
        'FLY': []
    }
    # solver = reaction_system(t,y)
    solver_rhs = reaction_system(p)
    # 可能需要添加第一步的时间要到e-10的量级
    sol = solve_ivp(
        solver_rhs,
        (t_eval[0], t_eval[-1]), # 设置步长范围
        y0,                      # 设置ODE方程
        t_eval = t_eval,
        method = 'LSODA',          # 设置计算方法 ‘BDF’ 'LSODA'
        rtol = 1e-5,
        atol = 1e-7,
        first_step = 1e-10
    )



    Y = sol.y.T
    time_log = np.log10(sol.t*1e3) # 以毫秒作单位

    obs = dict(
        FLY = p['sk']* k_1*(Y[:,I['x2']]+Y[:,I['x6']]
                                +Y[:,I['y2']]+Y[:,I['y6']]
                                +Y[:,I['z2']]+Y[:,I['z6']]
                                +Y[:,I['g2']]+Y[:,I['g6']]),
        # pott = 200*pt_arr,
        # pHn  = -np.log10(0.001*Hn_arr)*0.4,
        # pHp  = -np.log10(0.001*Hp_arr)*0.4,
        # dpH  = (-np.log10(0.001*Hn_arr)*0.4
        #         - (-np.log10(0.001*Hp_arr)*0.4)),
        # xz2  = 1.8e9*(Y[:,I['x2']]+Y[:,I['y2']]+Y[:,I['g2']]+Y[:,I['z2']])/1.62,
        # xz6  = 1.8e9*(Y[:,I['x6']]+Y[:,I['y6']]+Y[:,I['g6']]+Y[:,I['z6']])/1.62,
        # xz4  = 500*(Y[:,I['x4']]+Y[:,I['y4']]+Y[:,I['g4']]+Y[:,I['z4']])/1.62,
        # xz5  = 500*(Y[:,I['x5']]+Y[:,I['y5']]+Y[:,I['g5']]+Y[:,I['z5']])/1.62,
        # xz45 = (500*(Y[:,I['x4']]+Y[:,I['y4']]+Y[:,I['g4']]+Y[:,I['z4']])
        #         +500*(Y[:,I['x5']]+Y[:,I['y5']]+Y[:,I['g5']]+Y[:,I['z5']]))/1.62,
        # yy1  = 1000*Y[:,I['y1']]/1.62,
        # yy17 = 1000*(Y[:,I['y1']]+Y[:,I['y2']]+Y[:,I['y3']]+Y[:,I['y4']]
        #              +Y[:,I['y5']]+Y[:,I['y6']]+Y[:,I['y7']])/1.62,
        # pqhh = Y[:,I['PQH2']]/(Y[:,I['PQH2']]+Y[:,I['PQ']]+1e-12),
        # xz3  = 4.5e6*(Y[:,I['x3']]+Y[:,I['y3']]+Y[:,I['g3']]+Y[:,I['z3']])/1.62,
        # xz7  = 4.5e6*(Y[:,I['x7']]+Y[:,I['y7']]+Y[:,I['g7']]+Y[:,I['z7']])/1.62,
        # xz347 = 40*np.array(p680_hist)/1.62,
        # vwoc = 100*(V_hist[:,11]+V_hist[:,4]+V_hist[:,18]+V_hist[:,31]),
        # vv41 = 10*V_hist[:,41],
        # xg4  = 40*(Y[:,I['x4']]+Y[:,I['g4']])/1.62,
        # gg5  = 7050*Y[:,I['g5']]/1.62,
        # yy5  = 7050*Y[:,I['y5']]/1.62,
        # zz5  = 7050*Y[:,I['z5']]/1.62,
        # xx5  = 7050*Y[:,I['x5']]/1.62,
    )

    # 储存FLY
    df = pd.DataFrame({
        'time': sol.t,
        'FLY': obs['FLY']
    }).dropna()
    # 作图

    plt.figure(figsize=(12, 6))
    # plt.plot(experimental_times, experimental_value, 'o', label='Experiment', markersize=6, alpha=0.7)
    plt.plot(t_eval, obs['FLY'], '-', label='Model', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('FLY Intensity')
    plt.title('Model Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_plot.png', dpi=300)
    plt.show()

    
    return df
results_df = simulate_system()

# 保存数据到CSV
results_df.to_csv('simulation_results.csv', index=False)
print("Simulation completed. Results saved to simulation_results.csv")
