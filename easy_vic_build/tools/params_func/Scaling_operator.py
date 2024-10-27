# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
from collections import Counter
from functools import reduce


def multiply(x, y):
    return x * y


class Scaling_operator:
    #TODO nonlinear scaling operator

    @staticmethod
    def Harmonic_mean(data):
        data = np.array(data)
        return len(data) / np.sum(1/data)
    
    @staticmethod
    def Arithmetic_mean(data):
        data = np.array(data)
        return np.mean(data)

    @staticmethod
    def Geometric_mean(data):
        data = np.array(data)
        return pow(reduce(multiply, data), 1/len(data))
    
    @staticmethod
    def Maximum_difference(data):
        data = np.array(data)
        return max(data) - min(data)
    
    @staticmethod
    def Majority(data):
        data = np.array(data)
        counter = Counter(data)
        return max(counter.keys(), key=counter.get)
    

if __name__ == "__main__":
    x = np.array([2, 3])
    so = Scaling_operator()
    so.Arithmetic_mean(x)
    so.Geometric_mean(x)
    so.Harmonic_mean(x)