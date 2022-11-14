import sys
import math
import time
arguments = sys.argv

levels = []
files = ['spiral','circle','gaussian','xor']


def read_results():
    results = {}
    results_counts = {}
    train_file = "results.csv"
    with open(train_file,'r') as i:
        lines = i.readlines()
    for it in lines:
        cont = it.replace("\n","").split(",")
        score = float(cont[1])
        if "circle" in cont[0]:
            key = cont[0].replace("circle-","")
        elif "gaussian":
            key = cont[0].replace("gaussian-","")
            score *=2
        elif "xor":
            score *=1.5
            key = cont[0].replace("xor-","")
        elif "spiral":
            key = cont[0].replace("spiral-","")
        if key in results:
            results[key] += score
            results_counts[key] += 1
        else:
            results[key] = score
            results_counts[key] = 1

    text_file = open("agregated.csv", "w")
    for it in results:
        val = it+","+str(results[it]/results_counts[key])+","+str(results_counts[key])+"\n"
        n = text_file.write(val)
    text_file.close()
read_results()
