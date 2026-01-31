import os
import numpy as np
import skrf as rf  # 用于读取 .sNp 文件

class ReadResults:
    def __init__(self, file_path):
        """
        初始化 ReadResults 类。
        :param file_path: 文件路径（支持 .sNp 和 .txt 文件）
        """
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()  # 获取文件扩展名

    def read_snp(self):
        """
        读取 .sNp 文件。
        :return: 频率（Hz）和 S 参数（复数形式）
        """
        if self.file_extension not in ['.s1p', '.s2p', '.s3p', '.s4p']:
            raise ValueError("文件扩展名不是 .sNp 格式！")

        network = rf.Network(self.file_path)  # 读取 .sNp 文件
        frequencies = network.f  # 频率（Hz）
        s_params = network.s  # S 参数（复数形式）

        return frequencies, s_params

    def read_txt(self):
        """
        读取 .txt 文件。
        :return: 数据（NumPy 数组）
        """
        if self.file_extension != '.txt':
            raise ValueError("文件扩展名不是 .txt 格式！")

        data = np.genfromtxt(self.file_path, skip_header=2)  # 读取 .txt 文件，从第三行开始是数据
        #data = np.genfromtxt(self.file_path, skip_header=2, usecols=(0, 1))
        return data

    def read(self):
        """
        根据文件扩展名自动调用相应的读取方法。
        :return: 文件数据
        """
        if self.file_extension in ['.s1p', '.s2p', '.s3p', '.s4p']:
            return self.read_snp()
        elif self.file_extension == '.txt':
            return self.read_txt()
        else:
            raise ValueError("不支持的文件格式！")