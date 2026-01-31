from LHSsample_file import LHSample as LHS
import random
import scipy.io as scio
import time
import numpy as np
from function_call_CST_parallel import call_CST
from function_call_CST import call_CST1
import os

def feasibility_mask(X):
    """
    返回可行性 mask：g1,g2,g3,g4 全部 < 0 则为 True
    """
    g1 = 0.9 - (0.5 - X[:, 5]) * X[:, 0]
    g2 = 0.9 - (0.5 - X[:, 6]) * X[:, 0]
    g3 = 0.9 - 0.5 * X[:, 0] * X[:, 4]
    g4 = 0.45 + (X[:, 4] - 1) * X[:, 0] / 2
    g5 = 0.45 - X[:, 0] * X[:, 5]
    g6 = 0.45 - X[:, 0] * X[:, 6]
    g7 = 1.8 - X[:, 7] * X[:, 1]
    g8 = 0.45 + (X[:, 3] - 1) * X[:, 1] / 2

    return (g1 < 0) & (g2 < 0) & (g3 < 0) & (g4 < 0) & (g5 < 0) & (g6 < 0) & (g7 < 0) & (g8 < 0)

def generate_feasible_LHS(dim_opti, bounds, target_n, batch_n=48, max_rounds=200):
    """
    循环 LHS 采样，直到收集到 target_n 个可行样本（或达到 max_rounds）
    batch_n: 每轮先生成多少个候选点
    """
    feasible_list = []
    total_gen = 0

    for r in range(1, max_rounds + 1):
        X_cand = LHS(dim_opti, bounds, batch_n)
        total_gen += X_cand.shape[0]

        mask = feasibility_mask(X_cand)
        X_feas = X_cand[mask]

        if X_feas.size > 0:
            feasible_list.append(X_feas)

        cur = sum(a.shape[0] for a in feasible_list)
        print(f"[Feasible筛选] round={r}, 本轮可行={X_feas.shape[0]}, 累计可行={cur}, 累计生成={total_gen}")

        if cur >= target_n:
            X_all = np.vstack(feasible_list)
            return X_all[:target_n]

    # 到这里说明约束太严格/采样不够，仍没凑够
    if feasible_list:
        X_all = np.vstack(feasible_list)
        print(f"警告：达到 max_rounds={max_rounds} 仍不足 target_n={target_n}，实际可行={X_all.shape[0]}，将用现有可行样本继续。")
        return X_all
    else:
        print(f"警告：达到 max_rounds={max_rounds} 仍无可行样本，将直接返回空数组。")
        return np.empty((0, dim_opti))

if __name__ == '__main__':
    # 记录总运行开始时间
    total_start_time = time.perf_counter()

    soft_path_1 = r"D:\Program Files (x86)\CST Studio Suite 2022\AMD64/CST DESIGN ENVIRONMENT_AMD64.exe"
    model_path = r"D:\lushengjie\antenna1\patch_8_21_good1.cst"
    fullname_temp = rf"D:\lushengjie\antenna1\temp_train"
    fullname_temp_filed = rf"D:\lushengjie\antenna1\temp_train_filed"

    design_point = np.array([[4.44, 16, 0.45, 0.92, 0.8, 0.25, 0.11, 0.21]])

    bounds = np.array([
        [3, 6],  # w1
        [14, 17],  # l
        [0.3, 0.6],  # g
        [0.75, 0.95],  # w
        [0.55, 0.85],  # s
        [0.1, 0.3],  # d2
        [0.1, 0.3],  # d1
        [0.1, 0.22]
    ])

    dim_opti = 8
    pop_coarse = 48
    pop_fine = 4

    name_output = [r"1D Results\S-Parameters\S1,1", r"Tables\1D Results\Directivity,Theta=0,Phi=0.0",
                   r"Tables\1D Results\Realized Gain,3D,Max. Value (Solid Angle)"]
    farfield_num = 0  # 远场方向图的个数

    # 初始样本点采集
    X = generate_feasible_LHS(
        dim_opti=dim_opti,
        bounds=bounds,
        target_n=pop_coarse,
        batch_n=pop_coarse,  # 每轮先生成 pop_coarse 个候选（也可以更大，比如 2*pop_coarse）
        max_rounds=200
    )

    if X.shape[0] == 0:
        raise RuntimeError("没有生成任何满足约束的样本，检查约束公式/变量含义或放宽采样策略。")

    para_select_name = ["w1", "l", "g", "ro1", "ro2", "ro3", "ro4", "ro5"]
    dataNew = fullname_temp + '\\X.mat'
    scio.savemat(dataNew, {'X': X})

    dataNew = fullname_temp + '\\bounds.mat'
    scio.savemat(dataNew, {'bounds': bounds})

    dataNew = fullname_temp + '\\design_point.mat'
    scio.savemat(dataNew, {'design_point': design_point})

    # 调用 CST
    initial_time = call_CST(X, para_select_name, dim_opti, X.shape[0], model_path, fullname_temp,
                           fullname_temp_filed, name_output, farfield_num, soft_path_1, max_workers=4)

    # ===== 新增代码：检查并重试缺失样本 =====
    required_files = [f"Spara_{i}.s1p" for i in
                      range(X.shape[0])]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(fullname_temp_filed, f))]
    if missing_files:
        print(f"首次运行后缺失 {len(missing_files)} 个样本文件，正在重试...")
        missing_indices = [int(f.split('_')[1].split('.')[0]) for f in missing_files]
        X_missing = X[missing_indices]

        # 传入原始索引作为 output_indices 参数
        call_CST1(X_missing, para_select_name, dim_opti, len(missing_indices),
                  model_path, fullname_temp, fullname_temp_filed,
                  name_output, farfield_num, soft_path_1,
                  max_workers=1, output_indices=missing_indices)

        # 再次检查是否还有缺失
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(fullname_temp_filed, f))]
        if missing_files:
            print(f"警告：仍有 {len(missing_files)} 个样本未完成，请手动检查")
        else:
            print("所有样本已成功完成！")

    print("每个任务的运行时间:", initial_time)
    # 记录总运行结束时间
    total_end_time = time.perf_counter()
    # 计算总运行时间
    total_elapsed_time = total_end_time - total_start_time
    print("总运行时间:", total_elapsed_time, "秒")

    # 读取结果
    import os
    from function_read_results import ReadResults as RR

    S_dB_all = []
    S_f_all = []
    AR_all = []
    AR_f_all = []
    Gain_all = []
    Gain_f_all = []

    for i in range(pop_coarse):
        # 读取S参数
        s_file = os.path.join(fullname_temp_filed, f"Spara_{i}.s1p")
        reader = RR(s_file)
        S_f, S = reader.read()
        S_dB = 20 * np.log10(np.abs(S.squeeze()))  # 转换为dB

        # 读取AR
        ar_file = os.path.join(fullname_temp_filed, f"AR_{i}.txt")
        reader = RR(ar_file)
        AR_data = reader.read()
        AR_f, AR = AR_data[:, 0], AR_data[:, 1]

        # 读取Gain
        gain_file = os.path.join(fullname_temp_filed, f"Gain_{i}.txt")
        reader = RR(gain_file)
        Gain_data = reader.read()
        Gain_f, Gain = Gain_data[:, 0], Gain_data[:, 1]

        # 将当前样本数据添加到列表
        S_dB_all.append(S_dB)
        S_f_all.append(S_f)
        AR_all.append(AR)
        AR_f_all.append(AR_f)
        Gain_all.append(Gain)
        Gain_f_all.append(Gain_f)

    # 转换为NumPy数组
    S_dB_all = np.array(S_dB_all)
    S_f_all = np.array(S_f_all)
    AR_all = np.array(AR_all)
    AR_f_all = np.array(AR_f_all)
    Gain_all = np.array(Gain_all)
    Gain_f_all = np.array(Gain_f_all)

    # 保存所有样本数据到.mat文件
    scio.savemat(os.path.join(fullname_temp, 'S_dB.mat'), {'S11_dB': S_dB_all})
    scio.savemat(os.path.join(fullname_temp, 'S_f.mat'), {'S_f': S_f_all})
    scio.savemat(os.path.join(fullname_temp, 'G_all.mat'), {'Gain_all': Gain_all})
    scio.savemat(os.path.join(fullname_temp, 'Gain_f.mat'), {'Gain_f': Gain_f_all})