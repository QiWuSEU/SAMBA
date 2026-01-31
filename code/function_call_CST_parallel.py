import os
from subprocess import run
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from CSTScriptGenerator import CSTScriptGenerator
import numpy as np

import os
import shutil
import time
from subprocess import run
from CSTScriptGenerator import CSTScriptGenerator

def run_cst_simulation(i, X, para_select_name, dim_opti, model_path, fullname_temp,
                       fullname_temp_filed, name_output, farfield_num, soft_path_1, max_workers):

    start0 = time.perf_counter()
    # 1. 使用唯一工程文件名避免并发冲突
    save_path = os.path.join(fullname_temp, f"project_{i}.cst")

    # 若不存在则复制模型
    if not os.path.exists(save_path):
        shutil.copy(model_path, save_path)

    # 2. 构造导出文件路径（建议添加正确扩展名）
    S_Param_path = os.path.join(fullname_temp_filed, f"Spara_{i}")
    S_Param_temp = name_output[0]

    AR_path = os.path.join(fullname_temp_filed, f"AR_{i}.txt")
    AR_temp = name_output[1]

    Gain_path = os.path.join(fullname_temp_filed, f"Gain_{i}.txt")
    Gain_temp = name_output[2]

    # 3. 初始化宏脚本生成器
    script_generator = CSTScriptGenerator(f'main_{i}.bas')

    script_generator.write_begin(save_path, save_path)

    # 写入参数
    for j in range(dim_opti):
        script_generator.write_para(para_select_name[j], X[i, j])

    # 写入求解器调用指令（含 Rebuild + Solver.Start）
    script_generator.write_simulation()

    # 写入结果导出指令
    script_generator.write_Touchstone(S_Param_path, S_Param_temp)
    script_generator.write_post(AR_temp, AR_path)
    script_generator.write_post(Gain_temp, Gain_path)

    # 方向图导出
    for p in range(farfield_num):
        Farfield_path = os.path.join(fullname_temp_filed, f"Farfield_{p}_{i}.txt")
        Farfield_temp = name_output[2 + p]
        script_generator.write_Farfield(Farfield_temp, Farfield_path)

    script_generator.write_end()

    # 4. 运行 CST 宏
    process = run(f'"{soft_path_1}" -m main_{i}.bas', capture_output=True, text=True)

    # 5. 仿真状态检查
    if process.returncode == 0:
        print(f"任务 {i} 运行成功！")
    else:
        print(f" 任务 {i} 运行失败")
        print("CST 输出错误：", process.stderr)

    elapsed = time.perf_counter() - start0
    return elapsed


def call_CST(X, para_select_name, dim_opti, pop_opti, model_path, fullname_temp,
             fullname_temp_filed, name_output, farfield_num, soft_path_1, max_workers=4):
    """
    并行调用 CST 运行仿真。
    :param X: 参数矩阵
    :param para_select_name: 参数名称列表
    :param dim_opti: 参数维度
    :param pop_opti: 样本数量
    :param model_path: 原始 CST 模型路径
    :param fullname_temp: 临时文件夹路径
    :param fullname_temp_filed: 结果文件夹路径
    :param name_output: 输出项名称列表
    :param farfield_num: 方向图数量
    :param soft_path_1: CST 可执行文件路径
    :param max_workers: 最大并行任务数
    :return: 每个任务的运行时间列表
    """
    initial_time = []

    # 创建临时文件夹和结果文件夹
    os.makedirs(fullname_temp, exist_ok=True)
    os.makedirs(fullname_temp_filed, exist_ok=True)

    # 使用 ProcessPoolExecutor 并行运行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_cst_simulation, i, X, para_select_name, dim_opti, model_path, fullname_temp,
                fullname_temp_filed, name_output, farfield_num, soft_path_1, max_workers
            )
            for i in range(pop_opti)
        ]

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                elapsed = future.result()
                initial_time.append(elapsed)
            except Exception as e:
                print(f"任务运行失败: {e}")

    return initial_time

