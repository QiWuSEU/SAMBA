# 读取结果
import os
from function_read_results import ReadResults as RR
import numpy as np
import scipy.io as scio
def score_real(gain, s11, target_Gain=5.2, target_S11=-10):
    """
    真实样本打分规则：
    - 满足两个约束：1.0
    - S11 满足，Gain 轻微违反：0.9
    - Gain 满足，S11 轻微违反：0.7
    - Gain 和 S11 都轻微违反：0.5
    - 其余（任一严重违反）：0.0
    """
    gain_ok = gain >= target_Gain
    s11_ok = s11 <= target_S11
    gain_soft = target_Gain - 0.1 <= gain < target_Gain
    s11_soft = -10 < s11 < -9

    if gain_ok and s11_ok:
        return 1.0
    elif s11_ok and gain_soft:
        return 0.9
    elif gain_ok and s11_soft:
        return 0.7
    elif gain_soft and s11_soft:
        return 0.5
    else:
        return 0.0

def area_calculate(X):
    """面积计算"""
    area = X[1]*X[3]
    return area

def read_simulation(X,fullname_temp_filed, fullname_temp,target_Gain, target_S11):
    """读取仿真数据"""
    S_dB_all = []
    S_f_all = []
    Gain_all = []
    Gain_f_all = []
    for i in range(X.shape[0]):
        """每次取1个样本"""
        # 读取S参数
        s_file = os.path.join(fullname_temp_filed, f"Spara_{i}.s1p")
        reader = RR(s_file)
        S_f, S = reader.read()
        S_dB = 20 * np.log10(np.abs(S.squeeze()))  # 转换为dB

        # 读取Gain
        gain_file = os.path.join(fullname_temp_filed, f"Gain_{i}.txt")
        reader = RR(gain_file)
        Gain_data = reader.read()
        Gain_f, Gain = Gain_data[:, 0], Gain_data[:, 1]

        # 收集数据,将一维向量存起来
        S_dB_all.append(S_dB)
        S_f_all.append(S_f)
        Gain_all.append(Gain)
        Gain_f_all.append(Gain_f)
    # 转为数组
    S_dB_all = np.array(S_dB_all)
    S_f_all = np.array(S_f_all)
    Gain_all = np.array(Gain_all)
    Gain_f_all = np.array(Gain_f_all)

    return Gain_all, Gain_f_all, S_dB_all, S_f_all


def extract_gain_s11_at_targets(
    X,
    fullname_temp_filed,
    f_target_gain,
    f_target_s11,
    *,
    s_prefix="Spara_",
    s_ext=".s1p",
    g_prefix="Gain_",
    g_ext=".txt",
    skip_bad=True,
    verbose=False,
):
    """
    逐样本读取 S11 曲线与 Gain 曲线，并在目标频点处取最近点的标量值。
    返回：
        cand_gain: (n_valid,)  每个有效样本在 f_target_gain 处的增益
        cand_s11 : (n_valid,)  每个有效样本在 f_target_s11 处的 S11(dB)
        valid_ids: (n_valid,)  有效样本的原始索引
        bad_infos: list[(i, err)]  失败样本信息
    """
    cand_gain = []
    cand_s11  = []
    valid_ids = []
    bad_infos = []

    for i in range(X.shape[0]):
        try:
            # --- 读 S11 ---
            s_file = os.path.join(fullname_temp_filed, f"{s_prefix}{i}{s_ext}")
            reader = RR(s_file)
            S_f, S = reader.read()
            S = np.asarray(S).squeeze()
            S_f = np.asarray(S_f).squeeze()

            # 转 dB
            S_dB = 20.0 * np.log10(np.abs(S))

            # 找最近频点
            idx_s = int(np.abs(S_f - f_target_s11).argmin())
            s11_at = float(S_dB[idx_s])

            # --- 读 Gain ---
            g_file = os.path.join(fullname_temp_filed, f"{g_prefix}{i}{g_ext}")
            reader = RR(g_file)
            G_data = reader.read()   # 期望 (Ng,2)
            G_data = np.asarray(G_data)

            if G_data.ndim != 2 or G_data.shape[1] < 2:
                raise ValueError(f"Gain file has invalid shape: {G_data.shape}")

            G_f = G_data[:, 0].astype(float)
            G   = G_data[:, 1].astype(float)

            idx_g = int(np.abs(G_f - f_target_gain).argmin())
            gain_at = float(G[idx_g])

            # --- 记录 ---
            cand_s11.append(s11_at)
            cand_gain.append(gain_at)
            valid_ids.append(i)

            if verbose:
                print(f"[{i}] S11@{S_f[idx_s]:.6g}={s11_at:.3f} dB, "
                      f"Gain@{G_f[idx_g]:.6g}={gain_at:.3f} dBi")

        except Exception as e:
            bad_infos.append((i, str(e)))
            if not skip_bad:
                raise
            continue

    cand_gain = np.asarray(cand_gain, dtype=float)
    cand_s11  = np.asarray(cand_s11, dtype=float)
    valid_ids = np.asarray(valid_ids, dtype=int)
    return cand_gain, cand_s11, valid_ids, bad_infos
