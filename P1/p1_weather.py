#!/usr/bin/env python
# coding: utf-8

# In[128]:


# return the Manhattan distance between two dictionary data points from the data set.
def manhattan_distance(data_point1, data_point2):
    diffTMAX = abs(data_point1['TMAX'] - data_point2['TMAX'])
    diffPRCP = abs(data_point1['PRCP'] - data_point2['PRCP'])
    diffTMIN = abs(data_point1['TMIN'] - data_point2['TMIN'])
    return diffTMAX + diffPRCP + diffTMIN
    
#return a list of data point dictionaries read from the specified file.
def read_dataset(filename):
    file = open(filename)
    res = list()
    for line in file:
        data = line.split()
        dic = {}
        dic['DATE'] = data[0]
        dic['TMAX'] = float(data[2])
        dic['PRCP'] = float(data[1])
        dic['TMIN'] = float(data[3])
        dic['RAIN'] = data[4]
        res.append(dic)
            
    file.close()
    return res


#return a prediction of whether it is raining or not based on a majority vote of the list of neighbors.
def majority_vote(nearest_neighbors):
    cntT = 0
    cntF = 0
    for neighbor in nearest_neighbors:
        if neighbor['RAIN'] == 'TRUE':
            cntT = cntT + 1
        else:
            cntF = cntF + 1
    if cntT >= cntF:
        return 'TRUE'
    else:
        return 'FALSE'
    
#using the above functions, return the majority vote prediction for whether it's raining or not on the provided test point.   
def k_nearest_neighbors(filename, test_point, k, year_interval):
    allLis = read_dataset(filename)
    validLis = list()
    #get the test year and its left and right intervaL
    pointYear = getYear(test_point)
    leftYear = pointYear - (year_interval - 1)
    rightYear = pointYear + (year_interval - 1)
    
    #get all valid dates in the interval
    for element in allLis:
        if getYear(element) >= leftYear and getYear(element) <= rightYear:
            validLis.append(element)
        
    #If there isn't any valid neighbor, default to 'TRUE' 
    if len(validLis) == 0:
        return 'TRUE'
    
    #sort those days by its distance to test point
    for validEle in validLis:
        validEle['DIST'] = manhattan_distance(test_point, (validEle))
    
    validLis.sort(key = getDis)

    forVote = []
    #If the number of the valid neighbors is smaller than the input value k, keep as many valid neighbors as possible.
    if(len(validLis) < k+1):
        for x in range (len(validLis)+1):
            forVote.append(validLis[x])
        return majority_vote(forVote)
    else:   #get the first kth element to vote
        for x in range (k+1):
            forVote.append(validLis[x])
        return majority_vote(forVote)
#get the year in the each row of the file  
def getYear(data):
    date_entries = data["DATE"].split("-")
    return int(date_entries[0])

def getDis(e):
    return e["DIST"]


# In[ ]:





# In[ ]:




