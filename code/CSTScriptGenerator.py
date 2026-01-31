class CSTScriptGenerator:
    def __init__(self, script_name='main.bas'):
        """
        初始化脚本生成器。
        :param script_name: 生成的 VBA 脚本文件名，默认为 'main.bas'
        """
        self.script_name = script_name

    def write_begin(self, model_path, save_path):
        """
        写入脚本的起始部分，包括打开模型和保存模型。
        :param model_path: CST 模型文件路径
        :param save_path: 临时保存路径
        """
        with open(self.script_name, 'w') as fn:
            fn.write(f'''
    '#Language "WWB-COM"
    Option Explicit
    Sub Main
      'Starts CST Studio Suite
    OpenFile ("{model_path}")
    SaveAs ("{save_path}", {True})
''')

    def write_para(self, para_select_name, X):
        """
        写入参数设置部分。
        :param para_select_name: 参数名称
        :param X: 参数值
        """
        with open(self.script_name, 'a') as fn:
            fn.write(f'''
    StoreParameter("{para_select_name}", {X})
''')

    def write_simulation(self):
        """
        写入仿真部分，包括重建模型和启动求解器。
        """
        with open(self.script_name, 'a') as fn:
            fn.write(f'''
    Rebuild
    Solver.Start
''')

    def write_Touchstone(self, S_Param_path, S_Param_temp):
        with open(self.script_name, 'a') as fn:
            fn.write(f'''
    SelectTreeItem ("{S_Param_temp}")
    With TOUCHSTONE
        .Reset
        .FileName ("{S_Param_path}")
        .Impedance (50)
        .FrequencyRange ("Full")
        .Renormalize (True)
        .UseARResults (False)
        .SetNSamples (401)
        .Write
    End With
''')

    def write_Farfield(self, Farfield_temp, Farfield_path):
        """
        写入 Farfield 导出功能。
        :param Farfield_temp: Farfield 树节点路径
        :param Farfield_path: Farfield 文件导出路径
        """
        with open(self.script_name, 'a') as fn:
            fn.write(f'''
    SelectTreeItem ("{Farfield_temp}")
    With ASCIIExport
        .Reset
        .FileName ("{Farfield_path}")
        .Mode ("FixedNumber")
        .Execute
    End With
''')

    def write_post(self, Post_temp, Post_path):
        """
        写入后处理结果导出功能。
        :param Post_temp: 后处理树节点路径
        :param Post_path: 后处理文件导出路径
        """
        with open(self.script_name, 'a') as fn:
            fn.write(f'''
    SelectTreeItem ("{Post_temp}")
    With ASCIIExport
        .Reset
        .FileName ("{Post_path}")
        .Mode ("FixedNumber")
        .Execute
    End With
''')

    def write_end(self):
        """
        写入脚本的结束部分。
        """
        with open(self.script_name, 'a') as fn:
            fn.write(f'''
End Sub
''')