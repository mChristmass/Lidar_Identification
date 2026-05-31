import numpy as np
from fontTools.misc.cython import returns
import os
import re
import math
# 适配单通道归一化，即输入参数是单通道的（双通道的在Unet1中）
def data_normalization(npy_path):
    data = np.load(npy_path)

    # 归一化：归一为0~1
    data_valid = data >0
    data_max = data.max()
    data_min = data[data_valid].min()
    norm_data = np.where(data >0, (data -data_min ) /(data_max -data_min ) +0.1 ,0) # 防止出现0，因为要与原生的0区分开
    new_data = np.reshape(norm_data,[norm_data.shape[0],1,128,128])

    return new_data