from scipy.linalg import eigh  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#load the dataset from a provided .npy file, 
#re-center it around the origin and return it as a NumPy array of floats
def load_and_center_dataset(filename):
    x = np.load(filename)
    x = np.reshape(x,(2000,784))
    np.mean(x, axis=0)
    x = x - np.mean(x, axis=0)
    return x

#calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
def get_covariance(dataset):
    x = np.dot(np.transpose(dataset),np.array(dataset))
    np.mean(x, axis=0) 
    x = x / 1999
    return x
    
# — perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) 
#with the largest m eigenvalues on the diagonal,
#and a matrix (NumPy array) with the corresponding eigenvectors as columns
def get_eig(S, m):
    Lambda,vector = eigh(np.array(S))
    res = np.zeros([m, m])
    U = np.empty([len(vector), m])
    for i in range(m):
        res[i, i] = Lambda[len(Lambda) - i - 1]
        U[:, i] = vector[:, len(vector) - i - 1]
    return res, U

    
#similar to get_eig, but instead of returning the first m, return all eigenvectors that explains more than perc % of variance
def get_eig_perc(S, perc):
    Lambda,vector = eigh(S)
    valid_count = 0
    for i in range(len(S)):
        cur = Lambda[len(Lambda ) - i - 1]
        if cur/np.sum(Lambda) > perc:
            valid_count += 1
    res = np.zeros([valid_count, valid_count])
    U = np.empty([len(vector), valid_count])
    for i in range(valid_count):
        cur = Lambda[len(Lambda ) - i - 1]
        if cur/np.sum(Lambda) > perc:
           res[i, i] = Lambda[len(Lambda) - i - 1]
           U[:, i] = vector[:, len(vector) - i - 1]
    return res, U

#project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
def project_image(image, U):
    m = len(U[0])
    project = np.zeros([len(U), 1])
    for i in range(m):
        eng_vector = []
        for j in range(len(U)):
            eng_vector.append(U[j, i])
        product = np.dot(np.transpose(eng_vector), image)
        project[:, 0] += product * U[:, i]
    res = np.transpose(project)[0]
    return res

#— use matplotlib to display a visual representation of the original image
#and the projected image side-by-side
def display_image(orig, proj): 
    x = np.reshape(orig,(28,28))   
    y = np.reshape(proj,(28, 28))
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,3))
    img_orig = ax1.imshow(x, aspect='equal',cmap='gray')
    ax1.set_title('Original')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(img_orig, cax=cax)
    img_proj = ax2.imshow(y, aspect='equal',cmap='gray')
    ax2.set_title('Projection')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(img_proj, cax=cax)
    plt.tight_layout()
    plt.show()

from scipy.linalg import eigh  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#load the dataset from a provided .npy file, 
#re-center it around the origin and return it as a NumPy array of floats
def load_and_center_dataset(filename):
    x = np.load(filename)
    x = np.reshape(x,(2000,784))
    np.mean(x, axis=0)
    x = x - np.mean(x, axis=0)
    return x

#calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
def get_covariance(dataset):
    x = np.dot(np.transpose(dataset),np.array(dataset))
    np.mean(x, axis=0) 
    x = x / 1999
    return x
    
# — perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) 
#with the largest m eigenvalues on the diagonal,
#and a matrix (NumPy array) with the corresponding eigenvectors as columns
def get_eig(S, m):
    Lambda,vector = eigh(np.array(S))
    res = np.zeros([m, m])
    U = np.empty([len(vector), m])
    for i in range(m):
        res[i, i] = Lambda[len(Lambda) - i - 1]
        U[:, i] = vector[:, len(vector) - i - 1]
    return res, U

    
#similar to get_eig, but instead of returning the first m, return all eigenvectors that explains more than perc % of variance
def get_eig_perc(S, perc):
    Lambda,vector = eigh(S)
    valid_count = 0
    for i in range(len(S)):
        cur = Lambda[len(Lambda ) - i - 1]
        if cur/np.sum(Lambda) > perc:
            valid_count += 1
    res = np.zeros([valid_count, valid_count])
    U = np.empty([len(vector), valid_count])
    for i in range(valid_count):
        cur = Lambda[len(Lambda ) - i - 1]
        if cur/np.sum(Lambda) > perc:
           res[i, i] = Lambda[len(Lambda) - i - 1]
           U[:, i] = vector[:, len(vector) - i - 1]
    return res, U

#project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
def project_image(image, U):
    m = len(U[0])
    project = np.zeros([len(U), 1])
    for i in range(m):
        eng_vector = []
        for j in range(len(U)):
            eng_vector.append(U[j, i])
        product = np.dot(np.transpose(eng_vector), image)
        project[:, 0] += product * U[:, i]
    res = np.transpose(project)[0]
    return res

#— use matplotlib to display a visual representation of the original image
#and the projected image side-by-side
def display_image(orig, proj): 
    x = np.reshape(orig,(28,28))   
    y = np.reshape(proj,(28, 28))
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,3))
    img_orig = ax1.imshow(x, aspect='equal',cmap='gray')
    ax1.set_title('Original')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(img_orig, cax=cax)
    img_proj = ax2.imshow(y, aspect='equal',cmap='gray')
    ax2.set_title('Projection')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(img_proj, cax=cax)
    plt.tight_layout()
    plt.show()

