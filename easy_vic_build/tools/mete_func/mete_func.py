# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np


def cal_VP(T, RH):
    # T in C
    es = 6.112 * np.exp((17.67*T)/(T+243.5))
    e = RH * es / 100
    return e