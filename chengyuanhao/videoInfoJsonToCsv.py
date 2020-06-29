import json
import csv
import codecs
import re
import pandas as pd

def transJsonToCsv(jsonList, csvPath):
    dfData = []
    for j_obj in jsonList:
    #read the file
        print(j_obj)
        jsonData= json.loads(j_obj)
        label = list(jsonData.keys())
        # label = ['name', 'end', 'start', 'text', 'id']
        if 'end' not in label:
            continue


        #extract the data
        jsonData_name = jsonData['name']
        jsonData_end = jsonData['end']
        #the type of jsonData_end & jsonData_start & jsonData_text is list
        jsonData_start = jsonData['start']
        jsonData_text = jsonData['text']
        jsonData_id = jsonData['id']


        # to ensure that the data in line with norms
        assert len(jsonData_end) == len(jsonData_start) == len(jsonData_text) and len(label) == 5, 'the palce which has an error'

        jsonData_subtitle = list(zip(jsonData_text, jsonData_start, jsonData_end))

        # create the dataframe
        dfData.append({'name': jsonData_name, 'id':jsonData_id, 'subtitle': jsonData_subtitle})
        
    file = open(csvPath, "w", encoding='utf-8')
    df = pd.DataFrame(dfData,columns=['name', 'id', 'subtitle'])
    df.to_csv(file)
    file.close()


def getJsonList(jsonPath):
    with open(jsonPath,'r',encoding='utf-8') as f:
        f.seek(0)
        res = f.read()

    #use regularExpression to extract
    regularExpression = re.compile(r'[{]("name".*?")[}]', re.S)
    jsonWithNoBracketLsit = re.findall(regularExpression, res)
    jsonList = []
    for item in jsonWithNoBracketLsit:
        # print(item)
        if item.startswith('"name"'):
            itemNew = '{'+ item +'}'
            jsonList.append(itemNew)
    # print(jsonList)
    #each element in jsonList is a json
    return jsonList
    


if __name__ == '__main__':
    jsonPath = './jsonToCsv/video_info.json'
    csvPath = './jsonToCsv/video_info.csv'
    jsonList= getJsonList(jsonPath)
    transJsonToCsv(jsonList, csvPath)
