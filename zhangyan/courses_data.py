import json
import csv
import codecs


if __name__ == '__main__':

    jsonData = codecs.open( "Track1/course_info.json", 'r')
    csvfile = open('course_info' + '.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    flag = True

    for line in jsonData:
        dic = json.loads(line[0:-1])
        if flag:
            keys = ['course_id','video_id']
            writer.writerow(keys)  # 将属性列表写入csv中
            flag = False
        value = list(dic.values())

        # 读取json数据的每一行，将values数据一次一行的写入csv中
        for str in value[0][:]:
            if(str[0] == 'P'):
                 value[0].remove(str);

        writer.writerow(list(reversed(list(value))))

    jsonData.close()
    csvfile.close()