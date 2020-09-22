import csv
# train.csv
# 0:id,
# 1:course_id,
# 2:video_id,
# 3:watching_count,
# 4:video_duration,
# 5:local_watching_time,
# 6:video_progress_time,
# 7:video_start_time,
# 8:video_end_time,
# 9:local_start_time,
# 10:local_end_time,
# 11:drop

# 0:id,
# 1:course_id,
# 2:video_id,
# 3:watching_count,
# 4:video_duration,
# 5:local_watching_time,
# 6:video_progress_time,
# 7:video_start_time,
# 8:video_end_time,
# 9:local_start_time,
# 10:local_end_time,
# 11:drop
def Data_Processing(csv_in_adrs,flag = 1):                #flag为1则归一化,默认为0
    with open(csv_in_adrs,"r",encoding="utf-8") as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    ID = [data[i][0] for i in range(1,len(data))]                    #ID
    Watching_Count = [float(data[i][3]) for i in range(1,len(data))]               #观看次数
    Local_Watching_Time = [float(data[i][5]) for i in range(1,len(data))]          #观看时长
    Video_Time = [(float(data[i][8])- float(data[i][7])) for i in range(1,len(data))]     #时间点差
    Local_Time = [(float(data[i][10])- float(data[i][9])) for i in range(1,len(data))]   #日期差
    Video_Progress_Time = [float(data[i][6]) for i in range(1,len(data))]             #考虑了倍速的播放时长
    if(flag == 1):          # 归一化操作
        maxtemp1 = 0
        maxtemp2 = 0
        maxtemp3 = 0
        maxtemp4 = 0
        maxtemp5 = 0
        for i in range(len(Watching_Count)):
            if(maxtemp1 < Watching_Count[i]):
                maxtemp1 = Watching_Count[i]
            if (maxtemp2 < Local_Watching_Time[i]):
                maxtemp2 = Local_Watching_Time[i]
            if (maxtemp3 < Video_Time[i]):
                maxtemp3 = Video_Time[i]
            if (maxtemp4 < Local_Time[i]):
                maxtemp4 = Local_Time[i]
            if (maxtemp5 < Video_Progress_Time[i]):
                maxtemp5 = Video_Progress_Time[i]
        for i in range(len(ID)):
            Watching_Count[i] /= maxtemp1
            Local_Watching_Time[i] /= maxtemp2
            Video_Time[i] /= maxtemp3
            Local_Time[i] /= maxtemp4
            Video_Progress_Time[i] /= maxtemp5

    Data = list(zip(ID,Local_Watching_Time,Watching_Count,Local_Time,Video_Time,Video_Progress_Time))
    head = ["ID","Local_Watching_Time","Watching_Count","Local_Time","Video_Time","Video_Progress_Time"]

    with open("New_Processed_test.csv", "w", newline="", encoding="utf-8") as outfile:   #写CSV文件
        csvf = csv.writer(outfile)
        csvf.writerow(head)
        csvf.writerows(Data)

if __name__ == '__main__':
    csv_in_adrs = 'test.csv'     #train_data是训练集示例
    Data_Processing(csv_in_adrs)