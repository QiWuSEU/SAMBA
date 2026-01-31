from scipy.stats import norm
import numpy as np
from read_results import area_calculate
from sklearn.preprocessing import MinMaxScaler
import torch

def satisfy_con(s11_pred, gain_pred, s11_std, gain_std, target_S11, target_Gain):
    """计算可行性概率"""
    s11_prob = norm.cdf((target_S11 - s11_pred) / s11_std)
    gain_prob = 1 - norm.cdf((target_Gain - gain_pred) / gain_std)

    return s11_prob, gain_prob


def fitness_pred_min(x, gp_gain, gp_s11, penalty_min, target_Gain, target_S11, best_area, beta = 1):
    """注重面积"""
    fitness_pred_min_list = []
    for sample in x:
        # 数据预处理
        geom = sample.reshape(1, -1)
        geom_tensor = torch.tensor(geom, dtype=torch.double)

        # 分别使用两个独立的GPR模型进行预测
        with torch.no_grad():
            # 预测增益
            gp_gain.eval()
            posterior_gain = gp_gain.posterior(geom_tensor)
            gain_pred = posterior_gain.mean.item()  # 提取标量值
            gain_std = posterior_gain.variance.sqrt().item()

            # 预测S11
            gp_s11.eval()
            posterior_s11 = gp_s11.posterior(geom_tensor)
            s11_pred = posterior_s11.mean.item()
            s11_std = posterior_s11.variance.sqrt().item()

        # 计算约束满足的概率
        s11_prob, gain_prob = satisfy_con(s11_pred, gain_pred, s11_std, gain_std, target_S11, target_Gain)

        # 计算 s11_loss
        if s11_pred > target_S11:
            s11_loss = abs(s11_pred - target_S11) + (1 - s11_prob)
        else:
            s11_loss = (1 - s11_prob)

        # 计算 gain_loss
        if gain_pred < target_Gain:
            gain_loss = abs(target_Gain - gain_pred) + (1 - gain_prob)
        else:
            gain_loss = (1 - gain_prob)

        feasibility_threshold = 0.05  #可行性概率记入惩罚
        s11_lcb = s11_pred - beta * s11_std
        gain_ucb = gain_pred + beta * gain_std
        if (s11_prob * gain_prob <= feasibility_threshold) and (s11_lcb > target_S11) and (gain_ucb < target_Gain):
            penalty_factor = 1.0 + (1 - s11_prob * gain_prob) * 10  # 1-6倍
            curr_area = penalty_factor * area_calculate(sample)
            area_norm = curr_area / best_area
        else:
            curr_area = area_calculate(sample)
            area_norm = curr_area / best_area

        # 计算最终目标函数值
        fitness = area_norm + penalty_min * s11_loss + penalty_min * gain_loss
        fitness_pred_min_list.append(fitness)

    return np.array(fitness_pred_min_list)

def fitness_pred_max(x, gp_gain, gp_s11, penalty_max, target_Gain, target_S11, best_area, beta=0):
    """注重约束"""
    fitness_pred_max_list = []

    for sample in x:
        # 数据预处理
        geom = sample.reshape(1, -1)
        geom_tensor = torch.tensor(geom, dtype=torch.double)

        # 分别使用两个独立的GPR模型进行预测
        with torch.no_grad():
            # 预测增益
            gp_gain.eval()
            posterior_gain = gp_gain.posterior(geom_tensor)
            gain_pred = posterior_gain.mean.item()  # 提取标量值
            gain_std = posterior_gain.variance.sqrt().item()

            # 预测S11
            gp_s11.eval()
            posterior_s11 = gp_s11.posterior(geom_tensor)
            s11_pred = posterior_s11.mean.item()
            s11_std = posterior_s11.variance.sqrt().item()

        # 计算约束满足的概率
        s11_prob, gain_prob = satisfy_con(s11_pred, gain_pred, s11_std, gain_std, target_S11, target_Gain)
        curr_area = area_calculate(sample)
        # 计算 s11_loss
        if s11_pred > target_S11:
            s11_loss = abs(s11_pred - target_S11)/ abs(target_S11) + (1 - s11_prob)
        else:
            s11_loss = (1 - s11_prob)

        # 计算 gain_loss
        if gain_pred < target_Gain:
            gain_loss = abs(target_Gain - gain_pred)/ abs(target_Gain) + (1 - gain_prob)
        else:
            gain_loss = (1 - gain_prob)

        # 计算最终目标函数值（只包含约束惩罚项）
        base_fitness = penalty_max * s11_loss + penalty_max * gain_loss

        # 添加面积惩罚,一般不设置
        step = 0
        if curr_area >= (best_area-step):
            fitness = 1e12
        else:
            fitness = base_fitness

        fitness_pred_max_list.append(fitness)
    return np.array(fitness_pred_max_list)

def fitness_pred_mix(x, gp_gain, gp_s11, best_area, target_Gain, target_S11):
    """实际改进"""
    fitness_pred_mix_list = []
    for sample in x:
        geom = sample.reshape(1, -1)
        geom_tensor = torch.tensor(geom, dtype=torch.double)

        with torch.no_grad():
            # 预测增益
            gp_gain.eval()
            posterior_gain = gp_gain.posterior(geom_tensor)
            gain_pred = posterior_gain.mean.item()
            gain_std = posterior_gain.variance.sqrt().item()

            # 预测S11
            gp_s11.eval()
            posterior_s11 = gp_s11.posterior(geom_tensor)
            s11_pred = posterior_s11.mean.item()
            s11_std = posterior_s11.variance.sqrt().item()

        # 计算约束满足的概率
        s11_prob, gain_prob = satisfy_con(s11_pred, gain_pred, s11_std, gain_std, target_S11, target_Gain)
        curr_area = area_calculate(sample)
        feasibility = s11_prob * gain_prob

        step = 0
        if curr_area < (best_area - step):
            fitness = (curr_area-best_area) * feasibility
        else:
            fitness = 1e12
        fitness_pred_mix_list.append(fitness)

    return np.array(fitness_pred_mix_list)

def fitness_pred_gain(x, gp_gain, gp_s11, penalty_max, target_Gain, target_S11, best_area, beta=0):
    fitness_pred_gain_list = []
    fitness_pred_gain_std_list = []
    fitness_pf = []
    for sample in x:
        # 数据预处理
        geom = sample.reshape(1, -1)
        geom_tensor = torch.tensor(geom, dtype=torch.double)

        # 分别使用两个独立的GPR模型进行预测
        with torch.no_grad():
            # 预测增益
            gp_gain.eval()
            posterior_gain = gp_gain.posterior(geom_tensor)
            gain_pred = posterior_gain.mean.item()  # 提取标量值
            gain_std = posterior_gain.variance.sqrt().item()

            # 预测S11
            gp_s11.eval()
            posterior_s11 = gp_s11.posterior(geom_tensor)
            s11_pred = posterior_s11.mean.item()
            s11_std = posterior_s11.variance.sqrt().item()

        # 计算约束满足的概率
        s11_prob, gain_prob = satisfy_con(s11_pred, gain_pred, s11_std, gain_std, target_S11, target_Gain)
        pf = gain_prob

        fitness_pred_gain_list.append(gain_pred)
        fitness_pred_gain_std_list.append(gain_std)
        fitness_pf.append(pf)
    return np.array(fitness_pred_gain_list),np.array(fitness_pred_gain_std_list),np.array(fitness_pf)

def fitness_pred_s11(x, gp_gain, gp_s11, penalty_max, target_Gain, target_S11, best_area, beta=0):
    fitness_pred_s11_list = []
    fitness_pred_s11_std_list = []
    fitness_s11_pf = []
    for sample in x:
        # 数据预处理
        geom = sample.reshape(1, -1)
        #geom_norm = scaler.transform(geom)
        geom_tensor = torch.tensor(geom, dtype=torch.double)

        # 分别使用两个独立的GPR模型进行预测
        with torch.no_grad():
            # 预测增益
            gp_gain.eval()
            posterior_gain = gp_gain.posterior(geom_tensor)
            gain_pred = posterior_gain.mean.item()  # 提取标量值
            gain_std = posterior_gain.variance.sqrt().item()

            # 预测S11
            gp_s11.eval()
            posterior_s11 = gp_s11.posterior(geom_tensor)
            s11_pred = posterior_s11.mean.item()
            s11_std = posterior_s11.variance.sqrt().item()

        # 计算约束满足的概率
        s11_prob, gain_prob = satisfy_con(s11_pred, gain_pred, s11_std, gain_std, target_S11, target_Gain)
        pf = s11_prob

        fitness_pred_s11_list.append(s11_pred)
        fitness_pred_s11_std_list.append(s11_std)
        fitness_s11_pf.append(pf)
    return np.array(fitness_pred_s11_list),np.array(fitness_pred_s11_std_list),np.array(fitness_s11_pf)

