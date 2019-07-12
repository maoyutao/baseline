import sys
import os
import csv
import re

all_py_files = []

def searchDirFile(rootDir):
    global all_py_files
    for dir_or_file in os.listdir(rootDir):
        filePath = os.path.join(rootDir, dir_or_file)
        # 判断是否为文件
        if os.path.isfile(filePath):
            # 如果是文件再判断是否以.py结尾，不是则跳过本次循环
            if os.path.basename(filePath).endswith('.py'):
                all_py_files.append(filePath)
            else:
                continue
        # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
        elif os.path.isdir(filePath):
            searchDirFile(filePath)
        
path = '../public/py_github'
searchDirFile(path)

with open("py.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    for i in all_py_files:
        try:
            text = open(i).read()
            text = re.sub(r'[\n\r/,]', ' ', text)
            writer.writerow([i[len(path):], text])
        except KeyboardInterrupt:
            break
        except:
            print("error", i)