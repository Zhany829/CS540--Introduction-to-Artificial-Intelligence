#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import heapq
from _overlapped import NULL
# given a state of the puzzle, represented as a single list of integers with a 0 in the empty space, 
# print to the console all of the possible successor states
def print_succ(state):
    successors = get_successor(state)
    for i in range(len(successors)):
        h_value = get_Manhattan_dis(successors[i], [1,2,3,4,5,6,7,8,0])
        successor = successors[i]
        print(str(successor) + str(' h=') + str(h_value))

#recursive way to trace back to get the path from goal state to original state
def get_parent(info):
    #base case:the first state is met
    if info[4] == -1:
        print(f'{info[1]} h={info[3]} move: {info[2]}')
        return
    #recursion to trace back 
    else:
        get_parent(info[4])
        print(f'{info[1]} h={info[3]} move: {info[2]}')
#given a state of the puzzle, perform the A* search algorithm and print the path from the current state to the goal state
def solve(cur_state):
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    closed = []# store visited state
    open_pq = []#process pop and push
    g_score = 0# store the g_score
    state_info = [0,0,0,0,0]#set the first node
    state_info[0]=g_score+get_Manhattan_dis(cur_state,goal_state)
    state_info[1]= cur_state
    state_info[2]= g_score #number of moves
    state_info[3]= get_Manhattan_dis(cur_state,goal_state)
    state_info[4]= -1#parent index
    min_item = state_info#get the item with lowest score
    while not (min_item) == NULL:
        if cur_state == goal_state:#if reach the goal, print the path
            get_parent(min_item)
            return
        #get successors
        successors = get_successor(cur_state)
        g_score = min_item[2] + 1 # increase g score by one for every move
        for suc_state in successors:
            #generate the information for the state
            state_info = [0,0,0,0,0]
            state_info[0] = g_score+get_Manhattan_dis(suc_state,goal_state)
            state_info[1] = suc_state
            state_info[2] = g_score
            state_info[3] = get_Manhattan_dis(suc_state,goal_state)
            state_info[4] = min_item
            heapq.heappush(open_pq, state_info)
        
        min_item = heapq.heappop(open_pq)#pop the item with lowest score
        cur_state = min_item[1]
        if cur_state in closed:#remove visited state
            min_item = heapq.heappop(open_pq)#update the min_item
            cur_state = min_item[1]
        closed.append(cur_state)# put visited state into close


  
#author: Ajinkya Sonawane
#source:https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288
#The following code is obtained the inspiraion from source above.

#Generate successor nodes from the given node by moving the blank space
def get_successor(state):
    #get the position of empty place
    row,column = findEmpty(state,0)
    #neighbor_list contains position values for moving the empty place in either of
    #the 4 directions [up,down,left,right] respectively.
    neighbor_list = [[row,column-1],[row,column+1],[row-1,column],[row+1,column]]
    valid_succ = []
    for i in neighbor_list:
        successors = move(state,row,column,i[0],i[1])
        if successors is not None:
            successor = []
            #generate the valid successor
            for j in range(3):
                for k in range(3):
                    successor.append(successors[j][k])
            valid_succ.append(successor)
    sorted_succ = sorted(valid_succ)
    return sorted_succ
        
def move(state,x1,y1,x2,y2):
    # Move the empty place in the given direction and if the position value are out
    # of limits the return None
    if x2 >= 0 and x2 < 3 and y2 >= 0 and y2 < 3:
        successor = []
        successor = copy(state)
        temp = successor[x2][y2]
        successor[x2][y2] = successor[x1][y1]
        successor[x1][y1] = temp
        return successor
    else:
        return None
            
#Copy function to create a similar matrix of the given node
def copy(state):
    ret = []
    for i in range(3):
        ret.append([0 for j in range(3)])
    index = 0
    for i in range(3):
        for j in range(3):
            ret[i][j] = state[index]
            index += 1
    return ret
    
    
#find the empty place in the current mtrix
def findEmpty(state,empty):
        temp = copy(state)
        #Specifically used to find the position of the blank space 
        for i in range(3):
            for j in range(3):
                if temp[i][j] == empty:
                    return i,j
                
#author: AcrobatAHA
#source:https://github.com/AcrobatAHA/How-to-solve-an-8-puzzle-problem-using-A-Algorithm-in-python-/blob/master/Heuristics%20for%20the%208-puzzle.py
#The following code is obtained the inspiraion from source above.

#get the Manhattan distance
def get_Manhattan_dis(oneD_cur_state, oneD_goal_state):
    dis = 0
    cur_state = copy(oneD_cur_state)
    goal_state = copy(oneD_goal_state)
    for i in range(len(cur_state)):
        for j in range(len(cur_state)):
            if cur_state[i][j] == 0:
                continue
            elif (goal_state[0][0] == cur_state[i][j]):
                dis += (abs(i-0) + abs(j-0))
            elif (goal_state[0][1] == cur_state[i][j]):
                dis += (abs(i-0) + abs(j-1))
            elif (goal_state[0][2] == cur_state[i][j]):
                dis += (abs(i-0) + abs(j-2))
            elif (goal_state[1][0] == cur_state[i][j]):
                dis += (abs(i-1) + abs(j-0))            
            elif (goal_state[1][1] == cur_state[i][j]):
                dis += (abs(i-1) + abs(j-1))
            elif (goal_state[1][2] == cur_state[i][j]):
                dis += (abs(i-1) + abs(j-2))
            elif (goal_state[2][0] == cur_state[i][j]):
                dis += (abs(i-2) + abs(j-0))
            elif (goal_state[2][1] == cur_state[i][j]):
                dis += (abs(i-2) + abs(j-1))   
            elif (goal_state[2][2] == cur_state[i][j]):
                dis += (abs(i-2) + abs(j-2))
    return dis




# In[ ]:





# In[ ]:




