import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average=2.0):
        self.y_estimate = y_initial_measure
        # Code
        self.num_average = int(num_average)  # 평균 계산에 사용할 샘플 수
        self.window = [y_initial_measure]    # 초기 측정값 저장
        self.y_estimate = y_initial_measure  # 초기 추정값
        
    def estimate(self, y_measure):
        # Code
        self.window.append(y_measure)
        if len(self.window) > self.num_average:
            self.window.pop(0)  # window size 초과 시 가장 오래된 값 제거
        self.y_estimate = sum(self.window) / len(self.window)

    
if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example_Filter_1.csv")      
    #signal = pd.read_csv("01_filter/Data/example_Filter_2.csv")

    y_estimate = MovingAverageFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i])
        signal.y_estimate[i] = y_estimate.y_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



