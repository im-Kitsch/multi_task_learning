import numpy as np



class CodeTransfer:
    def __init__(self, str_code_path):
        unique_code = np.loadtxt(str_code_path, dtype="U8")
        num_code_list = np.arange(unique_code.shape[0])
        self.dict_code2str = dict(zip(num_code_list, unique_code))
        self.dict_str2code = dict(zip(unique_code, num_code_list))
        return

    def str2num(self, str_arr):
        return itemgetter(*str_arr.tolist())(self.dict_str2code)

    def num2str(self, num_arr):
        return itemgetter(*num_arr.tolist())(self.dict_code2str)
