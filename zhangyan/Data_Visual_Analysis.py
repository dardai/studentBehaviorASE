import csv

import matplotlib.pyplot as plt
import numpy as np


#plot根据列表绘制出有意义的图形，linewidth是图形线宽，可省略
def showplot(input_values,squares,flag0,flag1):
    plt.plot(input_values,squares,linewidth=5)
    #设置图标标题
    #plt.title("",fontsize = 24)
    #设置坐标轴标签
    switch0 = {1:'Local_Watching_time',2:'Watching_Count',3:'Local_Time',4:'Video_Time',5:'Video_Progress_Time'}
    if flag0 != 1:
        plt.xlabel(switch0[flag1],fontsize = 14)
    else:
        plt.xlabel('subtitles_'+str(flag1), fontsize=14)
    plt.ylabel("drop_ratio",fontsize = 14)
    #设置刻度标记的大小
    plt.tick_params(axis='both',labelsize = 14)
    #打开matplotlib查看器，并显示绘制图形
    plt.show()

def get_data_y():
    with open('train.csv',"r",encoding="utf-8") as infile:
        reader = csv.reader(infile)
        drop = [row[11] for row in reader]
    del(drop[0])
    print(drop)
    return drop

def get_data_x(flag0,flag1):
    if flag0 == 0:
        if flag1 > 5 or flag1 < 1:
            print("error!")
            return 0;

        with open('New_Processed_train.csv',"r",encoding="utf-8") as infile:
            reader = csv.reader(infile)
            sample = [row[flag1] for row in reader]
        del(sample[0])
        return sample
    else:
        if flag1 > 20 or flag1 < 1:
            print("error!")
            return 0;
        with open('final_train.csv', "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            sample = [row[flag1] for row in reader]
        del (sample[0])
        return sample;




def Analysis(flag0,flag1,n,t):   #n个区间，舍去t个以下的样本
    sample = get_data_x(flag0,flag1)
    drop = get_data_y()
    standard = np.linspace(0,1,num=n)
    print(standard)
    drop_ratio = np.zeros(n)
    drop_sum = np.zeros(n)
    drop_true = np.zeros(n)

    for i in range(len(sample)):
        position = search_position(sample[i],standard)
        drop_sum[position]+=1
        if drop[i] == '1':
            drop_true[position]+=1

    for i in range(n):
         if drop_sum[i] != 0 and drop_sum[i]>t:
            drop_ratio[i] = 1.0*drop_true[i]/drop_sum[i]

    print(drop_true)
    print(drop_sum)
    print(drop_ratio)
    showplot(standard,drop_ratio,flag0,flag1)

def search_position(key, standard):
    min = 0
    max = len(standard)-1
    while True:
        if(min == max):
            return max
        if float(key) <= float(standard[min]):
            return min
        else:
            min+=1




if __name__ == "__main__":

    #   Analysis 的 传入参数 flag0 和 flag1
    #   flag0 默认为 0
            #  flag0 flag1                  n           t
            #   0       1:  观看时长        区间个数      最小样本限制
            #   0       2:  观看次数
            #   0       3:  日期差
            #   0       4:  时间点差
            #   0       5:  播放时长(包括倍速)
            #   1       1-20:   20个字幕特征

    Analysis(1,20,36,5)
    print("Success")