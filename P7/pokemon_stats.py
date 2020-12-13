# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:57:55 2020

@author: yuzha
"""
import csv
import numpy as np
import math
import numpy as np
import scipy.cluster
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
#takes in a string with a path to a CSV file formatted as in the link above, 
#and returns the first 20 data points 
#(without the Generation and Legendary columns but retaining all other columns) 
#in a single structure.
def load_data(filepath):
   data = []
   with open(filepath,'r')as f:
       read = csv.reader(f)
       for index,info in enumerate(read):
           if index != 0 and index < 21:
            data.append(info[:11])   
   res = []
   dic = {}
   for i in range(len(data)):
        dic.update({'#': int(data[i][0])})
        dic.update({'Name': data[i][1]})
        dic.update({'Type 1': data[i][2]})
        dic.update({'Type 2': data[i][3]})
        dic.update({'Total': int(data[i][4])})
        dic.update({'HP': int(data[i][5])})
        dic.update({'Attack': int(data[i][6])})
        dic.update({'Defense': int(data[i][7])})
        dic.update({'Sp. Atk': int(data[i][8])})
        dic.update({'Sp. Def': int(data[i][9])})
        dic.update({'Speed': int(data[i][10])})
        res.append(dic)
        dic = {}
   return res

#takes in one row from the data loaded from the previous function, 
#calculates the corresponding x, y values for that Pokemon as specified above,
# and returns them in a single structure.
def calculate_x_y(stats):
    x = int(stats['Attack']) + int(stats['Sp. Atk']) + int(stats['Speed'])
    y = int(stats['Defense']) + int(stats['Sp. Def']) + int(stats['HP'])
    res = (x, y)
    return res

#performs single linkage hierarchical agglomerative clustering on the Pokemon
#with the (x,y) feature representation, and returns a data structure 
#representing the clustering.
def hac(dataset):
    m = len(dataset)
    res = []
    tree = []
    for data in dataset:
        tree.append([data])
    for i in range(m - 1):
        row = distance(tree)
        res.append(row)
        father = []
        for point1 in tree[row[0]]:
            father.append(point1)
        for point2 in tree[row[1]]:
            father.append(point2)
        tree.append(father)
        tree[row[0]] = tree[row[1]] = []
    return np.array(res)

   
def distance(list1):
    min_dis = 9999
    cur_dis = 0
    m = len(list1)
    for i in range(len(list1)):
        num1 = len(list1[i])
        for point1 in range(num1):
            x1 = list1[i][point1][0]
            y1 = list1[i][point1][1]
            for j in range(i + 1, m):
                num2 = len(list1[j])
                for point2 in range(num2):
                    x2 = list1[j][point2][0]
                    y2 = list1[j][point2][1]
                    cur_dis = euclidean_dis((x1,y1),(x2,y2))
                    if cur_dis < min_dis:
                        min_dis = cur_dis
                        index1 = min(i, j)
                        index2 = max(i, j)
                        nums_points = num1+num2
                    # tie breaking
                    elif cur_dis == min_dis:
                        cur_min_index = min(i, j)
                        cur_max_index = max(i, j)
                        if cur_min_index < index1:
                            min_dis = cur_dis
                            index1 = min(i, j)
                            index2 = max(i, j)
                            nums_points = num1 + num2
                        elif cur_min_index == index1:
                            if cur_max_index < index2:
                                min_dis = cur_dis
                                index1 = min(i, j)
                                index2 = max(i, j)
                                nums_points = num1+num2
    return [index1, index2, min_dis, nums_points]

def euclidean_dis(point1, point2):
    x = math.pow(abs(point1[0]-point2[0]), 2)
    y = math.pow(abs(point1[1]-point2[1]), 2)
    res = math.sqrt(x+y)
    return res
    






