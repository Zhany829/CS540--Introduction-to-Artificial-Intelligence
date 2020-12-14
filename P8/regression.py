import numpy as np
import random
import csv
from numpy import array
# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = None
    data = []
    with open(filename,'r')as f:
       read = csv.reader(f)
       for index,info in enumerate(read):
            data.append(info[1:]) 
    int_data = []
    for i in range(1,len(data)):
        int_data.append(data[i])
    for i in range(len(int_data)):
        for j in range(len(int_data[i])):
            int_data[i][j] = float(int_data[i][j])
    dataset = int_data
    return np.array(dataset)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    length = len(dataset)
    mean = 0
    sum_data = 0
    ele = []
    for i in range(length):
        sum_data += dataset[i][col]
        ele.append(dataset[i][col])
    mean = sum_data / length
    mean = round(mean,2)
    a = array(ele)
    sd = np.std(a)
    sd = round(sd,2)
    print(length)
    print(mean)
    print(sd)
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    length = len(dataset)
    sum = 0
    for i in range(len(dataset)):
        sum = betas[0] - dataset[i][0]
        for j in range(len(cols)):
            sum += betas[j+1] * dataset[i][cols[j]]
        sum = pow(sum,2)
        mse += sum
        sum = 0
        
    return mse/length
        
        


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    length = len(dataset)
    for z in range(len(cols)+1):
        sum = 0
        for i in range(len(dataset)):
            s = betas[0] - dataset[i][0]
            for j in range(len(cols)):
                s += betas[j+1] * dataset[i][cols[j]]
            if z == 0:
                sum += s
            else:
                sum += s * dataset[i][cols[z-1]]
        grads.append((sum*2/length))
        
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    result = []
    for i in range(1, T + 1):
        res = []
        res.append(i)
        for beta in betas:
            res.append(round(beta, 2))
        gd = gradient_descent(dataset, cols, betas)
        for j in range(len(betas)):
            betas[j] = betas[j] - eta * gd[j]
        mse = regression(dataset, cols, betas)
        res.append(round(mse, 2))
        result.append(res)
    for res in result:
        print(" ".join(map(str, res)))
    


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """  
    y = dataset[:, 0]
    columns = [0, *cols]
    feature = dataset[:, columns]
    for row in feature:
        row[0] = 1
    betas = []
    transpose_feature = np.transpose(feature) 
    betas = np.dot(np.dot(np.linalg.inv(np.dot(transpose_feature, feature)), transpose_feature), y)
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    ret = betas[1]
    for i in range(len(features)):
        ret += features[i] * betas[2 + i]
    return ret


def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)


def sgd(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.
    You must use random_index_generator() to select individual data points.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    random = random_index_generator(0,np.shape(dataset)[0],42)
    for t in range(1,T+1):
        random_next = next(random)
        betas_list = []
        for i in range(len(betas)):
            grads=[]
            data_cols = dataset[:,cols]
            for j in range(len(betas)):
                    s = betas[0]
                    for z in range(1,len(betas)):
                        s += betas[z] * data_cols[random_next][z-1]
                    s = (s - dataset[random_next][0]) * 2
                    if j != 0:
                        s = s * data_cols[random_next][j-1]
                    grads.append(s)
            betas_list.append(betas[i]-eta*(grads[i]))
        betas = betas_list
        res = ''
        for beta in betas:
            res+=str(round(beta,2))+' '
        print(str(t)+' '+str(round(regression(dataset,cols,betas),2))+' '+ res)



if __name__ == '__main__':
    pass
