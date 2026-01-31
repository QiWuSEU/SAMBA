from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor

from pymoo.core.problem import Problem  #从 pymoo 库中引入 Problem 类, 用于定义优化问题（如变量、目标函数、约束等）
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
import copy
import numpy as np
import scipy.io as scio
import matplotlib
import os
import time
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import csv

from function_call_CST_parallel import call_CST
from function_call_CST import call_CST1
from objective_functions import fitness_pred_min, fitness_pred_max, fitness_pred_mix, fitness_pred_gain, fitness_pred_s11
from LHSsample_file import LHSample as LHS
from read_results import read_simulation, score_real, area_calculate, extract_gain_s11_at_targets
import multiprocessing
from sklearn.preprocessing import MinMaxScaler

import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize  # 添加输入变换
from torch.optim import Adam  # 添加Adam优化器

def cache_position(gain, s11, area, score, sampler, i):
    """用Area与score比较出真的全局最优"""
    global cached_positions  # 使用全局变量
    global best_real_score  # 初始设为0
    global best_real_area  # 初始设为无穷大,这两个是全局变量,第-1次调用时,全局值被更新

    global best_real_position
    global best_real_gain
    global best_real_s11

    global area_history
    global score_history

    cached_positions.append((gain, s11, area, score, sampler))
    # 添加当前的分数和面积到历史列表
    score_history.append(score)
    area_history.append(area)

    if score > best_real_score or (score == best_real_score and area < best_real_area):  # 初始的best_score是0分,根据得分来
        """通过这个if判断选出符合题目要求的最优样本,能在这里直接输出吗"""
        best_real_position = sampler  # 最佳样本,满足条件就更新全局变量
        best_real_score = score  # 最佳分数,同时也是更新分数
        best_real_area = area  # 最佳面积，同时也是更新面积
        best_real_gain = gain
        best_real_s11 = s11
        print(f"目前是第：{i}轮BO的大循环")
        print(
            f"历史比较的最佳样本：{best_real_position};历史比较的最佳得分：{best_real_score};历史比较的最佳面积：{best_real_area}")
        print(
            f"历史比较的最佳样本的增益：{best_real_gain};历史比较的最佳样本的s11：{best_real_s11}")
    else:
        print(f"第{i}轮BO大循环没有改进")
        print(f"历史比较的最佳样本：{best_real_position};历史比较的最佳得分：{best_real_score};历史比较的最佳面积：{best_real_area}")
        print(f"历史比较的最佳样本的增益：{best_real_gain};历史比较的最佳样本的s11：{best_real_s11}")

def cache_plot_position(area,score):
    global area_plot_history
    global score_plot_history

    area_plot_history.append(area)
    score_plot_history.append(score)

def plot_iteration_vs_area_and_score1():
    # 创建图表（双y轴）
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制左侧的面积曲线
    ax1.plot(range(1, len(area_plot_history) + 1), area_plot_history,
             marker='o', color='b', label="Area")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Area', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 创建第二个y轴，共享x轴
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(score_plot_history) + 1), score_plot_history,
             marker='o', color='r', label="Score")
    ax2.set_ylabel('Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 设置图表标题和图例
    ax1.set_title('Iteration vs Area and Score')
    ax1.grid(True)

    # 显示图表
    plt.tight_layout()
    plt.show()

def check_constraints(X):
    """检查约束条件，返回满足约束的行的索引"""
    g1 = 0.9 - (0.5 - X[:, 5]) * X[:, 0]
    g2 = 0.9 - (0.5 - X[:, 6]) * X[:, 0]
    g3 = 0.9 - 0.5 * X[:, 0] * X[:, 4]
    g4 = 0.45 + (X[:, 4] - 1) * X[:, 0] / 2
    g5 = 0.45 - X[:, 0] * X[:, 5]
    g6 = 0.45 - X[:, 0] * X[:, 6]
    g7 = 1.8 - X[:, 7] * X[:, 1]
    g8 = 0.45 + (X[:, 3] - 1) * X[:, 1] / 2
    # 所有约束条件都要小于0
    valid_indices = np.where((g1 < 0) & (g2 < 0) & (g3 < 0) & (g4 < 0) & (g5 < 0) & (g6 < 0) & (g7 < 0) & (g8 < 0))[0]
    return valid_indices


cached_positions = []
best_real_score = 0
best_real_area = float('inf')
best_real_position = np.array([0, 0, 0, 0, 0, 0, 0, 0])
best_real_gain = 0
best_real_s11 = 0
area_history = []
score_history = []

area_plot_history = []
score_plot_history = []

def main():
    """主程序入口"""
    target_Gain = 5.1
    target_S11 = -10
    penalty_min = 0
    penalty_max = 1
    max_BO_iter = 30
    f_target = 1.88
    f_target_S_f = 1.88e9

    area_ini = 384  # 目前样本的面积和增益
    score_ini = 1
    gain_ini = 5.45
    s11_ini = -21.7
    sampler_ini = np.array([4.44, 16, 0.45, 0.92, 0.8, 0.25, 0.11, 0.21])
    ini = -2
    cache_plot_position(area_ini, score_ini)
    cache_position(gain_ini, s11_ini, area_ini, score_ini, sampler_ini, ini)

    #训练集数据读取
    data = scio.loadmat(r'D:\lushengjie\antenna1\train_8var_1_21.mat')
    total_samples = data['X'].shape[0]
    n_samples = 49  # 需要抽取的样本量
    fixed_index = 48
    print(f"固定包含的样本索引（Matlab第49行）: {fixed_index}")

    # 创建不包含固定索引的所有可能索引
    all_indices = np.arange(total_samples)
    remaining_indices = np.delete(all_indices, fixed_index)
    # 从剩余索引中随机抽取 n_samples - 1 个
    random_indices = np.random.choice(remaining_indices, size=n_samples - 1, replace=False)
    # 合并固定索引和随机索引
    sampled_indices = np.concatenate([[fixed_index], random_indices])
    print(f"选中的索引为{sampled_indices}")

    #对增益进行同步抽取
    Gain_all = data['Gain_all'][sampled_indices]
    print(Gain_all.shape)
    Gain_f = data['Gain_f'][sampled_indices]
    Gain_at_target = np.zeros(Gain_all.shape[0])
    freqs = Gain_f[0, :]
    idx = np.abs(freqs - f_target).argmin()
    # 提取所有样本在该频点的增益
    Gain_at_target = Gain_all[:, idx]
    print(f"训练集中距目标频点最近的索引为：{idx}")
    print(f"训练集中距目标频点最近的频点为：{freqs[idx]}")
    print(f"训练集中在该频点下的增益为:{Gain_at_target}")

    #对S11进行同步抽取
    S11_dB = data['S11_dB'][sampled_indices]
    S_f = data['S_f'][sampled_indices]
    S11_at_target = np.zeros(S11_dB.shape[0])
    freqs = S_f[0, :]
    idx = np.abs(freqs - f_target_S_f).argmin()
    # 提取所有样本在该频点的S11
    S11_at_target = S11_dB[:, idx]  # 取这频点的这1列所有样本值
    print(f"训练集中距目标频点最近的索引为：{idx}")
    print(f"训练集中距目标频点最近的频点为：{freqs[idx]}")
    print(f"训练集中在该频点下的S11为:{S11_at_target}")

    X_train = data['X'][sampled_indices]
    X_train_tensor = torch.tensor(X_train, dtype=torch.double)
    Y1 = torch.tensor(Gain_at_target, dtype=torch.double).unsqueeze(-1)
    Y2 = torch.tensor(S11_at_target, dtype=torch.double).unsqueeze(-1)

    # 定义明确的边界
    xl = torch.tensor([3, 14, 0.3, 0.75, 0.55, 0.1, 0.1, 0.1], dtype=torch.double)
    xu = torch.tensor([6, 17, 0.6, 0.95, 0.85, 0.3, 0.3, 0.22], dtype=torch.double)
    bounds = torch.stack([xl, xu])

    # 创建第一个GP（增益）
    gp1 = SingleTaskGP(
        train_X=X_train_tensor,
        train_Y=Y1,
        covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=9)),
        input_transform=Normalize(
            d=X_train_tensor.shape[-1],
            bounds=bounds,
            learn_bounds=False
        ),
        outcome_transform=Standardize(m=1),  # 输出标准化
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
    )

    # 创建第二个GP（S11）
    gp2 = SingleTaskGP(
        train_X=X_train_tensor,
        train_Y=Y2,
        covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=9)),
        input_transform=Normalize(
            d=X_train_tensor.shape[-1],
            bounds=bounds,
            learn_bounds=False
        ),
        outcome_transform=Standardize(m=1),
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
    )
    print("开始单独训练增益模型...")
    mll_gain = ExactMarginalLogLikelihood(gp1.likelihood, gp1)
    fit_gpytorch_mll_torch(
        mll_gain,
        optimizer=Adam(mll_gain.parameters(), lr=0.05),
        step_limit=100,
    )

    print("开始单独训练S11模型...")
    mll_s11 = ExactMarginalLogLikelihood(gp2.likelihood, gp2)
    fit_gpytorch_mll_torch(
        mll_s11,
        optimizer=Adam(mll_s11.parameters(), lr=0.05),
        step_limit=100,
    )

    print("两个独立模型训练完成！")
    scores = np.array([score_real(g, s) for g, s in zip(Gain_at_target, S11_at_target)])
    print("初始训练集样本评分：", scores)
    areas = np.array([area_calculate(x) for x in X_train])
    print("初始训练集样本面积：", areas)

    # 找到最大得分
    max_score = np.max(scores)
    best_indices = np.where(scores == max_score)[0]
    best_idx = best_indices[np.argmin(areas[best_indices])]

    # 最优样本信息
    X_best = X_train[best_idx]
    Gain_best = Gain_at_target[best_idx]
    S11_best = S11_at_target[best_idx]
    score_best = scores[best_idx]
    area_best = areas[best_idx]

    print("训练集中最优样本索引：", best_idx)
    print("训练集最优样本的参数：", X_best)
    print("训练集最优样本的增益：", Gain_best)
    print("训练集最优样本的S11：", S11_best)
    print("训练集最优样本的得分：", score_best)
    print("训练集最优样本的面积：", area_best)
    i_train = -1
    cache_plot_position(area_best, score_best) #存训练集最优样本
    cache_position(Gain_best, S11_best, area_best, score_best, X_best, i_train)

    class DirectConstraintSampling(Sampling):
        def __init__(self, check_constraints_func):
            super().__init__()
            self.check_constraints = check_constraints_func

        def _do(self, problem, n_samples, **kwargs):
            xl, xu = problem.bounds()
            bounds_array = np.column_stack([xl, xu])

            X_final = np.empty((0, problem.n_var))
            count = 0

            while len(X_final) < n_samples:
                remaining = n_samples - len(X_final)
                new_X = LHS(problem.n_var, bounds_array, max(remaining * 2, 10))
                valid_indices = self.check_constraints(new_X)
                valid_samples = new_X[valid_indices]

                if len(valid_samples) > 0:
                    X_final = np.vstack([X_final, valid_samples])
                    if len(X_final) > n_samples:
                        X_final = X_final[:n_samples]

                count += 1
                if count > 100:
                    break

            print(f"生成 {len(X_final)} 个约束样本，尝试了 {count} 次")
            print(f"生成边界的下界为：{xl},生成边界的上界为：{xu}")
            return X_final

    class MixedProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=8,
                n_obj=3,
                n_ieq_constr=9,
                n_eq_constr=0,
                xl=[3, 14, 0.3, 0.75, 0.55, 0.1, 0.1, 0.1],
                xu=[6, 17, 0.6, 0.95, 0.85, 0.3, 0.3, 0.22],
                vtype=["real", "real", "real", "real", "real", "real", "real", "real"]
            )
            self.GPR_gain = gp1
            self.GPR_s11 = gp2
            self.penalty_min = penalty_min
            self.penalty_max = penalty_max
            self.target_Gain = target_Gain
            self.target_S11 = target_S11
            self.best_area = area_ini

            self.area_factor = area_factor

        def _evaluate(self, X, out, *args, **kwargs):
            fitness_min = fitness_pred_min(X.copy(), self.GPR_gain, self.GPR_s11,self.penalty_min, self.target_Gain, self.target_S11,
                self.best_area)

            fitness_max = fitness_pred_max(X.copy(), self.GPR_gain, self.GPR_s11,self.penalty_max, self.target_Gain, self.target_S11,
                self.best_area
            )

            fitness_mix = fitness_pred_mix(X.copy(), self.GPR_gain, self.GPR_s11,self.best_area, self.target_Gain, self.target_S11)

            g1 = 0.9 - (0.5 - X[:, 5]) * X[:, 0]
            g2 = 0.9 - (0.5 - X[:, 6]) * X[:, 0]
            g3 = 0.9 - 0.5 * X[:, 0] * X[:, 4]
            g4 = 0.45 + (X[:, 4] - 1) * X[:, 0] / 2
            g5 = 0.45 - X[:, 0] * X[:, 5]
            g6 = 0.45 - X[:, 0] * X[:, 6]
            g7 = 1.8 - X[:, 7] * X[:, 1]
            g8 = 0.45 + (X[:, 3] - 1) * X[:, 1] / 2

            # ===== 动态面积约束 g9：用 self.area_factor 而不是写死 0.92 =====
            curr_area = X[:, 1] * (4 * X[:, 2] + 5 * X[:, 0])
            area_min = self.area_factor * self.best_area  # 希望 curr_area >= area_min
            g9 = area_min - curr_area  # g9 <= 0 ⇒ 满足面积下限

            out["F"] = np.column_stack([fitness_min, fitness_max, fitness_mix])
            out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9])

    area_factor = 0.9
    area_factor_min = 0.85
    area_factor_max = 0.99
    success_streak = 0
    fail_streak = 0
    save_dir = r"D:\lushengjie\antenna1\logs_best1" #BO每轮最优解存取位置

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "best_area_score.csv")
    for i in range(max_BO_iter):

        print(f"\n===== BO 第 {i + 1}/{max_BO_iter} 轮 =====")
        print(f"当前 area_factor = {area_factor:.3f}, 当前 area_ini(全局最优面积) = {area_ini:.4f}")

        t0_opt = time.perf_counter()
        problem = MixedProblem()
        constrained_sampling = DirectConstraintSampling(check_constraints)

        algorithm = NSGA2(
            pop_size=200,
            sampling=constrained_sampling,  #使用约束采样
            eliminate_duplicates=True
        )

        res = minimize(problem, algorithm, ("n_gen", 200), verbose=False)
        real_sampler = res.X

        fitness_cand_min = fitness_pred_min(real_sampler, gp1, gp2, penalty_min, target_Gain, target_S11, area_ini)
        print(f"只看面积的样本的适应度值为：{fitness_cand_min}")
        fitness_cand_max = fitness_pred_max(real_sampler, gp1, gp2, penalty_max, target_Gain, target_S11, area_ini)
        print(f"总体约束的样本的适应度值为：{fitness_cand_max}")
        fitness_cand_mix = fitness_pred_mix(real_sampler, gp1, gp2, area_ini, target_Gain, target_S11)
        print(f"混合面积和约束的样本的适应度值为：{fitness_cand_mix}")

        fitness_gain_pred, fitness_gain_std, fitness_gain_pf = fitness_pred_gain(real_sampler, gp1, gp2, penalty_max, target_Gain,
                                                                           target_S11, area_ini)
        print(f"样本预测增益均值为：{fitness_gain_pred}")
        print(f"样本预测增益方差为：{fitness_gain_std}")
        print(f"样本预测增益可行性概率为：{fitness_gain_pf}")
        fitness_s11_pred, fitness_s11_std, fitness_s11_pf = fitness_pred_s11(real_sampler, gp1, gp2, penalty_max,target_Gain,
                                                                            target_S11, area_ini)
        print(f"样本预测S11均值为：{fitness_s11_pred}")
        print(f"样本预测S11方差为：{fitness_s11_std}")
        print(f"样本预测S11可行性概率为：{fitness_s11_pf}")
        top_min = 10
        top_max = 10
        top_min_indices = np.argsort(fitness_cand_min)[:top_min]
        top_max_indices = np.argsort(fitness_cand_max)[:top_max]

        # 计算所有样本的面积
        all_areas = np.array([area_calculate(sample) for sample in real_sampler])

        print(f"当前最优面积: {area_ini:.3f}")

        print("聚类开始")
        kmeans = KMeans(n_clusters=4, random_state=42).fit(real_sampler)
        cand_16 = []

        for cluster in range(4):
            cluster_indices = np.where(kmeans.labels_ == cluster)[0]
            print(f"类别 {cluster} 包含 {len(cluster_indices)} 个样本")

            if len(cluster_indices) == 0:
                print(f"类别 {cluster} 为空，从备用池选2个")
                cluster_min = np.random.choice(top_min_indices, replace=False)
                cluster_max = np.random.choice(top_max_indices, replace=False)
                cand_16.append(real_sampler[cluster_min])
                cand_16.append(real_sampler[cluster_max])
                continue

            # 选择1: 该聚类中面积最小的点
            cluster_areas = all_areas[cluster_indices]
            min_area_idx = cluster_indices[np.argmin(cluster_areas)]
            cand_16.append(real_sampler[min_area_idx])
            print(f"  选择1 - 最小面积: {cluster_areas.min():.3f}")

            # 选择2: 该聚类中面积小于当前最优且适应度最小的点
            cluster_smaller_mask = cluster_areas < area_ini
            cluster_smaller_indices = cluster_indices[cluster_smaller_mask]

            if len(cluster_smaller_indices) > 0:
                smaller_fitness = fitness_cand_max[cluster_smaller_indices]
                best_small_idx = cluster_smaller_indices[np.argmin(smaller_fitness)]
                cand_16.append(real_sampler[best_small_idx])
                print(
                    f"  选择2 - 面积更小且适应度好: 面积={all_areas[best_small_idx]:.3f}, 适应度={fitness_cand_max[best_small_idx]:.3f}")
            else:
                # 没有面积小于当前最优的点，选择该聚类中适应度最小的点
                cluster_fitness = fitness_cand_max[cluster_indices]
                best_fitness_idx = cluster_indices[np.argmin(cluster_fitness)]
                cand_16.append(real_sampler[best_fitness_idx])
                print(
                    f"  选择2 - 聚类适应度最优: 面积={all_areas[best_fitness_idx]:.3f}, 适应度={fitness_cand_max[best_fitness_idx]:.3f}")

        print("选取的 16 个帕累托解为：")
        print(cand_16)
        length = len(cand_16)
        cand_16 = np.array(cand_16)

        t1_opt = time.perf_counter()
        opt_time = t1_opt - t0_opt
        print(f"选点/优化耗时: {opt_time:.3f} s")

        fitness_min_cand16 = fitness_pred_min(cand_16, gp1, gp2, penalty_min, target_Gain, target_S11, area_ini)
        fitness_max_cand16 = fitness_pred_max(cand_16, gp1, gp2, penalty_max, target_Gain, target_S11, area_ini)
        fitness_mix_cand16 = fitness_pred_mix(cand_16, gp1, gp2, area_ini, target_Gain, target_S11)
        fitness_gain_pred, fitness_gain_std, fitness_gain_pf = fitness_pred_gain(cand_16, gp1, gp2, penalty_max, target_Gain, target_S11, area_ini)
        fitness_s11_pred, fitness_s11_std, fitness_s11_pf = fitness_pred_s11(cand_16, gp1, gp2, penalty_max, target_Gain, target_S11, area_ini)
        for j in range(len(cand_16)):
            print(f"样本 {j}:")
            print(f"  fitness_min = {fitness_min_cand16[j]:.4f}")
            print(f"  fitness_max = {fitness_max_cand16[j]:.4f}")
            print(f"  fitness_mix = {fitness_mix_cand16[j]:.4f}")
            print(f"  fitness_gain_pred = {fitness_gain_pred[j]:.4f}")
            print(f"  fitness_gain_std = {fitness_gain_std[j]:.4f}")
            print(f"  fitness_gain_pf = {fitness_gain_pf[j]:.4f}")
            print(f"  fitness_s11_pred = {fitness_s11_pred[j]:.4f}")
            print(f"  fitness_s11_std = {fitness_s11_std[j]:.4f}")
            print(f"  fitness_s11_pf = {fitness_s11_pf[j]:.4f}")
            print("-" * 30)

        #CST软件位置;模型位置;临时模型和数据位置
        soft_path_1 = r"D:\Program Files (x86)\CST Studio Suite 2022\AMD64/CST DESIGN ENVIRONMENT_AMD64.exe"
        model_path = r"D:\lushengjie\antenna1\patch_8_21_good1.cst"
        fullname_temp = rf"D:\lushengjie\antenna1\temp_train5/temp_train_round{i}/"
        fullname_temp_filed = rf"D:\lushengjie\antenna1\temp_train_filed5/temp_train_filed_round{i}/"
        # 若目录不存在则创建
        os.makedirs(fullname_temp, exist_ok=True)
        os.makedirs(fullname_temp_filed, exist_ok=True)

        para_select_name = ["w1", "l", "g", "ro1", "ro2", "ro3", "ro4", "ro5"]
        name_output = [r"1D Results\S-Parameters\S1,1", r"Tables\1D Results\Directivity,Theta=0,Phi=0.0",
                       r"Tables\1D Results\Realized Gain,3D,Max. Value (Solid Angle)"]
        farfield_num = 0
        dim_opti = 8
        t0_sim = time.perf_counter()
        initial_time = call_CST(cand_16, para_select_name, dim_opti, cand_16.shape[0], model_path, fullname_temp,
                                fullname_temp_filed, name_output, farfield_num, soft_path_1, max_workers=4)

        # ===== 检查S参数文件是否缺失 =====
        required_files = [f"Spara_{i}.s1p" for i in
                          range(cand_16.shape[0])]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(fullname_temp_filed, f))]
        if missing_files:
            print(f"首次运行后缺失 {len(missing_files)} 个样本文件，正在重试...")
            missing_indices = [int(f.split('_')[1].split('.')[0]) for f in missing_files]
            X_missing = cand_16[missing_indices]

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

        # ===== 检查增益文件是否缺失 =====
        required_gain_files = [f"Gain_{i}.txt" for i in range(cand_16.shape[0])]
        missing_gain_files = [f for f in required_gain_files if
                              not os.path.exists(os.path.join(fullname_temp_filed, f))]

        if missing_gain_files:
            print(f"增益文件缺失 {len(missing_gain_files)} 个，正在处理...")
            missing_gain_indices = [int(f.split('_')[1].split('.')[0]) for f in missing_gain_files]
            X_missing_gain = cand_16[missing_gain_indices]

            # 重新计算缺失的增益
            call_CST1(X_missing_gain, para_select_name, dim_opti, len(missing_gain_indices),
                      model_path, fullname_temp, fullname_temp_filed,
                      name_output, farfield_num, soft_path_1,
                      max_workers=1, output_indices=missing_gain_indices)

            # 再次检查增益文件
            missing_gain_files = [f for f in required_gain_files if
                                  not os.path.exists(os.path.join(fullname_temp_filed, f))]
            if missing_gain_files:
                print(f"警告：仍有 {len(missing_gain_files)} 个增益文件未完成")

        t1_sim = time.perf_counter()
        sim_time = t1_sim - t0_sim
        print(f"仿真(含补跑/检查)耗时: {sim_time:.3f} s")

        t0_read = time.perf_counter()
        cand16_gain, cand16_s11, sim_valid_ids, bad_infos = extract_gain_s11_at_targets(
                                                                                        cand_16,
                                                                                        fullname_temp_filed,
                                                                                        f_target_gain=f_target,
                                                                                        f_target_s11=f_target_S_f,
                                                                                        verbose=True,
                                                                                        )

        cand_16_sim = cand_16[sim_valid_ids]
        print("valid:", sim_valid_ids)
        print("bad:", bad_infos)
        print("cand16_gain shape:", cand16_gain.shape)
        print("cand16_s11 shape:", cand16_s11.shape)

        new_gain = np.array(cand16_gain)
        new_s11 = np.array(cand16_s11)

        # 过滤掉增益为40的异常样本（连同对应的S11和输入特征一起筛掉）,仿真可能出现的问题
        valid_indices = np.where(new_gain != 40)[0]
        if len(valid_indices) < len(new_gain):
            removed_count = len(new_gain) - len(valid_indices)
            print(f"发现 {removed_count} 个增益为40的异常样本，已连同对应的S11和输入特征一起过滤")

            # 显示被过滤的样本信息
            removed_indices = np.where(new_gain == 40)[0]
            for idx in removed_indices:
                print(f"  移除样本 {idx}: 增益={new_gain[idx]}, S11={new_s11[idx]}")

        # 只保留有效样本（增益、S11、输入特征一起保留）
        valid_gain = new_gain[valid_indices]
        valid_s11 = new_s11[valid_indices]
        valid_cand_16 = cand_16_sim[valid_indices]
        print(f"过滤后有效样本数量: {len(valid_indices)}")

        # 更新训练数据集（只添加有效样本）
        if len(valid_indices) > 0:
            X_train = np.vstack((X_train, valid_cand_16))
            Gain_at_target = np.concatenate((Gain_at_target, valid_gain))
            S11_at_target = np.concatenate((S11_at_target, valid_s11))
            print(f"成功添加 {len(valid_indices)} 个有效样本到训练集")
        else:
            print("警告：本轮所有样本都被过滤，跳过更新训练数据")

        print(f"更新后训练数据形状: X={X_train.shape}")
        print(f"增益数据形状: {Gain_at_target.shape}")
        print(f"S11数据形状: {S11_at_target.shape}")

        cand16_score = []
        cand16_area = []
        for k in range(len(valid_gain)):
            score = score_real(valid_gain[k], valid_s11[k], target_Gain, target_S11)
            area = area_calculate(valid_cand_16[k])
            cand16_score.append(score)
            cand16_area.append(area)

        max_score = np.max(cand16_score)
        cand_16_score_new = np.array(cand16_score)
        high_score_indices = np.where(cand_16_score_new == max_score)[0]

        cand_16_area_new = np.array(cand16_area)
        cand_16best_index = high_score_indices[np.argmin(cand_16_area_new[high_score_indices])]

        cand_best_area = cand16_area[cand_16best_index]
        cand_best_score = cand16_score[cand_16best_index]

        # 判定本轮优化状态
        hard_success = (cand_best_score == 1.0) and (cand_best_area < area_ini)
        soft_success = (cand_best_score == 1.0) and (cand_best_area >= area_ini)
        failure = (cand_best_score != 1.0)

        # 更新 area_ini
        if hard_success:
            area_ini = cand_best_area
            print(f"【HARD SUCCESS】score=1 且面积更小，更新 area_ini = {area_ini:.4f}")
        elif soft_success:
            print(f"【SOFT SUCCESS】score=1，但面积 {cand_best_area:.4f} >= area_ini {area_ini:.4f}")
        else:
            print(f"【FAILURE】最佳样本分数为 {cand_best_score:.3f}，未达到1分")

        # 更新 streak
        if hard_success:
            success_streak += 1
            fail_streak = 0
        elif failure:
            fail_streak += 1
            success_streak = 0
        else:
            pass

        print(f"streaks: success_streak={success_streak}, fail_streak={fail_streak}")

        # 调整 area_factor
        if success_streak >= 2:
            old = area_factor
            area_factor = max(area_factor - 0.01, area_factor_min)
            success_streak = 0
            print(f"连续硬成功，减小 area_factor: {old:.3f} -> {area_factor:.3f}")

        if fail_streak >= 2:
            old = area_factor
            area_factor = min(area_factor + 0.02, area_factor_max)
            fail_streak = 0
            print(f"连续失败，增大 area_factor: {old:.3f} -> {area_factor:.3f}")

        cache_plot_position(cand_best_area, cand_best_score)

        for k in range(len(valid_gain)):
            cache_position(valid_gain[k], valid_s11[k], cand16_area[k], cand16_score[k], valid_cand_16[k], i)

        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.double)
        Y1_train_tensor = torch.tensor(Gain_at_target, dtype=torch.double).unsqueeze(-1)
        Y2_train_tensor = torch.tensor(S11_at_target, dtype=torch.double).unsqueeze(-1)

        gp1 = SingleTaskGP(
            train_X=X_train_tensor,
            train_Y=Y1_train_tensor,
            covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=X_train_tensor.shape[-1])),
            input_transform=Normalize(
                d=X_train_tensor.shape[-1],
                bounds=bounds,  # 显式指定边界
                learn_bounds=False  # 不自动学习边界，使用提供的边界
            ),
            outcome_transform=Standardize(m=1),
            likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
        )

        # 创建第二个GP（S11）
        gp2 = SingleTaskGP(
            train_X=X_train_tensor,
            train_Y=Y2_train_tensor,
            covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=X_train_tensor.shape[-1])),
            input_transform=Normalize(
                d=X_train_tensor.shape[-1],
                bounds=bounds,  # 显式指定边界
                learn_bounds=False  # 不自动学习边界，使用提供的边界
            ),
            outcome_transform=Standardize(m=1),
            likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
        )

        print("重新训练增益GPR模型...")
        mll_gain = ExactMarginalLogLikelihood(gp1.likelihood, gp1)
        optimizer_gain = Adam(mll_gain.parameters(), lr=0.05)
        fit_gpytorch_mll_torch(
            mll=mll_gain,
            optimizer=optimizer_gain,
            step_limit=100
        )

        print("重新训练S11 GPR模型...")
        mll_s11 = ExactMarginalLogLikelihood(gp2.likelihood, gp2)
        optimizer_s11 = Adam(mll_s11.parameters(), lr=0.05)
        fit_gpytorch_mll_torch(
            mll=mll_s11,
            optimizer=optimizer_s11,
            step_limit=100
        )

        print("两个独立GPR模型更新完成！")
        t1_read = time.perf_counter()
        read_time = t1_read - t0_read
        print(f"读取/过滤/更新训练集耗时: {read_time:.3f} s")

        time_total = opt_time + sim_time + read_time

        n_sim = int(len(cand_16))  # 本轮提交仿真的样本数
        best_idx_valid = int(cand_16best_index)

        row = {
            "iter": i,
            "best_area": float(cand_best_area),
            "best_score": float(cand_best_score),
            "best_idx_valid": int(best_idx_valid),
            "time_total_s": float(time_total),  # 总时间（秒）
            "opt_time_s": float(opt_time),
            "sim_time_s": float(sim_time),
            "read_time_s": float(read_time),
            "sim_count_s": float(len(cand_16)),
        }

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"已保存本轮 best_area/best_score/时间 到: {csv_path}")

    plot_iteration_vs_area_and_score1()
    print(f"总结最后的最好样本的分数为：{best_real_score}")
    print(f"总结最后的最好样本的面积为：{best_real_area}")
    pass

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()