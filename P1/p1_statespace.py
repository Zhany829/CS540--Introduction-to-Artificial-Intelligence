#!/usr/bin/env python
# coding: utf-8

# In[21]:

#returns a copy of state which fills the jug corresponding to 
#the index in which (0 or 1) to its maximum capacity. Do not modify state.
def fill(state, max, which):
    ret = [0,0]
    if which == 0:
        ret[0] = max[0]
        ret[1] = state[1]
    else:
        ret[0] = state[0]
        ret[1] = max[1]
    return ret
    
#returns a copy of state which empties the jug corresponding to the index 
#in which (0 or 1). Do not modify state.
def empty(state, max, which):
    ret = [0,0]
    if which == 0:
        ret[1] = state[1]
    else:
        ret[0] = state[0]
    return ret

#returns a copy of state which pours the contents of the jug at index source into the jug at index dest,
#until source is empty or dest is full. Do not modify state. xfer is shorthand for transfer.
def xfer(state, max, source, dest):
    ret = [0,0]
    ret[0] = state[0]
    ret[1] = state[1]
    capacityL = max[0]
    capacityR = max[1]
    curL = state[0]
    curR = state[1]
    if source == 0 and dest == 1:
        while curL > 0 and curR < capacityR:
            ret[0] = ret[0] - 1
            ret[1] = ret[1] + 1
            curL = curL - 1
            curR = curR + 1
    else:
        while curR > 0 and curL < capacityL:
            ret[0] = ret[0] + 1
            ret[1] = ret[1] - 1
            curL = curL + 1
            curR = curR - 1
    return ret

#prints the list of unique successor states of the current state in any order.
#This function will generate the unique successor states of the current state by applying fill, empty,
#xfer operations on the current state. (Note: You have to apply an operation at every step for generating a successor state.)
def succ(state, max):
    res = []
    res.append(fill(state,max,0))
    res.append(fill(state,max,1))
    res.append(empty(state,max,0))
    res.append(empty(state,max,1))
    res.append(xfer(state,max,1,0))
    res.append(xfer(state,max,0,1))
    #get rid of duplicate elements
    uniqLis = []
    for i in res:
        if not i in uniqLis:
            uniqLis.append(i)
    return uniqLis





