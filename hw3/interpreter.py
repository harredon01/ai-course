import sys
import math
import time
arguments = sys.argv

levels = []
files = ['spiral','circle','gaussian','xor']


def read_results(train_file):
    results = {}
    results_counts = {}

    with open(train_file,'r') as i:
        lines = i.readlines()
    for it in lines:
        item = {}
        cont = it.replace("\n","").split(",")
        score = float(cont[1])
        if "circle" in cont[0]:
            key = cont[0].replace("circle-","")
        elif "gaussian" in cont[0]:
            key = cont[0].replace("gaussian-","")
            score *=2
        elif "xor" in cont[0]:
            score *=1.5
            key = cont[0].replace("xor-","")
        elif "spiral" in cont[0]:
            key = cont[0].replace("spiral-","")
        if key in results:
            results[key] += score
            results_counts[key] += 1
        else:
            results[key] = score
            results_counts[key] = 1
    

    text_file = open("agregated.csv", "w")
    finals=[]
    for it in results:
        item = {}
        val = it+","+str(results[it]/results_counts[key])+","+str(results_counts[key])+"\n"
        #n = text_file.write(val)
        item['key'] = it
        item['score'] = results[it]
        item['avg']=results[it]/results_counts[key]
        item['counter'] = results_counts[it]
        finals.append(item)
    finals.sort(key=lambda x: x['avg'], reverse=False)
    for it in range(10):
        i = finals[it]
        print(i['key'],i['avg'],i['score'],i['counter'])
    text_file.close()

train_file = "results171.csv"
read_results(train_file)
