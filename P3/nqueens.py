#@author: Zhan Yu
import random
#given a state of the board, return a list of all valid successor states
def succ(state, static_x, static_y): 
    res = []
    if(not state[static_x] == static_y):
        return res
    size = len(state) #get the length of each side of the state board 
    #get the successors by moving the queen on the left of the static queen
    for i in range(0, static_x):
        if(state[i] - 1 >= 0):
            newState = state.copy()
            newState[i] -= 1
            res.append(newState)
        if(state[i] + 1 <= size):
            newState = state.copy()
            newState[i] += 1
            res.append(newState)
    #get the successors by moving the queen on the left of the static queen
    for i in range(static_x + 1, size):
        if(state[i] - 1 >= 0):
            newState = state.copy()
            newState[i] -= 1
            res.append(newState)
        if(state[i] + 1 < size):
            newState = state.copy()
            newState[i] += 1
            res.append(newState)
    return sorted(res)


#given a state of the board, return an integer score such that the goal state scores 0
def f(state):
    score = 0
    for i in range(len(state)):
        cur_score = score
        #check whether there is a conflict in the left of the queen
        for j in range(0, i):
            if(state[i] == state[j] or state[i] == state[j] + (i-j) or state[i] == state[j] - (i-j)):
                score += 1
                break
        if(cur_score == score - 1):
            continue
        #check whether there is a conflict in the right of the queen
        for j in range(i+1, len(state)):
            if(state[i] == state[j] or state[i] == state[j] + (j-i) or state[i] == state[j] - (j-i)):
                score += 1
                break
    return score


#given the current state, use succ() to generate the successors and return the selected next state
def choose_next(curr, static_x, static_y):
    if(succ(curr, static_x, static_y) == []):
        return
    successors = succ(curr, static_x, static_y)
    successors.append(curr)
    successors.sort()
    #print(successors)
    f_score = f(successors[0])
    res = successors[0]
    for i in range(len(successors)):
        if(f(successors[i]) < f_score):
            f_score = f(successors[i])
            res = successors[i]
    return res
      

#run the hill-climbing algorithm from a given initial state, return the convergence state
def n_queens(initial_state, static_x, static_y):
    print(initial_state,"- f={}".format(f(initial_state)))
    record_f_score = f(initial_state)
    cur_state = initial_state.copy()
    while((not f(choose_next(cur_state, static_x, static_y)) == record_f_score) and (not f(choose_next(cur_state, static_x, static_y))) == 0):
        cur_state = choose_next(cur_state, static_x, static_y)
        record_f_score = f(cur_state)
        print(cur_state,"- f={}".format(f(cur_state)))
    converge_state = choose_next(cur_state, static_x, static_y)
    print(converge_state,"- f={}".format(f(converge_state)))
    return converge_state
#run the hill-climbing algorithm on an n*n board with random restarts
def n_queens_restart(n, k, static_x, static_y):
    global_state = []
    local_states = {}
    random.seed(1)
    while(k > 0):
        #generate the random state
        random_state = []
        for i in range(n):
            random_num = random.randint(0, n)
            random_state.append(random_num)
        random_state[static_x] = static_y
        #get the most optimal condition for the random_state
        record_f_score = 0
        record_f_score = get_f(random_state)
        cur_state = random_state.copy()
        while((not get_f(choose_next(cur_state, static_x, static_y)) == record_f_score) and (not get_f(choose_next(cur_state, static_x, static_y))) == 0):
            cur_state = choose_next(cur_state, static_x, static_y)
            record_f_score = get_f(cur_state)
        optimal_state = choose_next(cur_state, static_x, static_y)
        optimal_f = get_f(choose_next(cur_state, static_x, static_y))
        if(optimal_f == 0):
            global_state.append(optimal_state)
            break
        else:
            key = tuple(optimal_state)
            local_states.update({key:optimal_f}) 
        k -= 1
        
    if(not global_state == []):
        print(global_state[0],"- f={}".format(get_f(global_state[0])))
    else:
        sort_orders = sorted(local_states.items(), key=lambda x: x[1])
        f = sort_orders[0][1]
        for i in range(0, len(sort_orders)):
            if(sort_orders[i][1] == f):
                print(list(sort_orders[i][0]),"- f={}".format(sort_orders[i][1]))
        

def get_f(state):
    score = 0
    for i in range(len(state)):
        cur_score = score
        #check whether there is a conflict in the left of the queen
        for j in range(0, i):
            if(state[i] == state[j] or state[i] == state[j] + (i-j) or state[i] == state[j] - (i-j)):
                score += 1
                break
        if(cur_score == score - 1):
            continue
        #check whether there is a conflict in the right of the queen
        for j in range(i+1, len(state)):
            if(state[i] == state[j] or state[i] == state[j] + (j-i) or state[i] == state[j] - (j-i)):
                score += 1
                break
    return score       


















