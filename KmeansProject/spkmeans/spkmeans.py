import math 
import sys
import pandas as pd
import numpy as np

import mykmeanssp as km
np.random.seed(0) 
def initCentros(allPoints, k):
    vectors = allPoints
    n, dim = vectors.shape
    first_index = int(np.random.choice(n))  
    mu1 = vectors[first_index]  
    centroids = np.zeros((k, dim))  
    centroids[0] = mu1  
    distances = np.zeros((k, n)) 
    for i in range(n):
        tmp_arr = np.power((vectors[i] - centroids[0]), 2)
        tmp_sum = 0
        for u in range(tmp_arr.shape[0]):
            tmp_sum += tmp_arr[u]
        distances[0][i] = math.sqrt(tmp_sum)
        

    returned_Indexes = np.zeros((k))  
    returned_Indexes[0] = first_index  

    z = 1
    while z < k:  
        d_i = np.min(distances[:z, ], axis=0)
        probs = [0 for c in range(n)]
        summ = d_i.sum()
        for r in range(n):
            probs[r] = d_i[r] / summ
        rand_indx = np.random.choice(n, p=probs)
        returned_Indexes[z] = rand_indx
        centroids[z] = vectors[rand_indx]
        for i in range(n):
            tmp_arr = np.power((vectors[i] - centroids[z]), 2)
            tmp_sum = 0
            for u in range(tmp_arr.shape[0]):
                tmp_sum += tmp_arr[u]
            distances[z][i] = math.sqrt(tmp_sum)
        z += 1
    vectorsList = vectors.tolist()
    centrosList = centroids.tolist()
    indxArrList = returned_Indexes.tolist()
    intindexes = []  
    for index in indxArrList:  
        intindexes.append(int(index))
    result =[]
    result.append(vectorsList)
    result.append(centrosList)
    result.append(indxArrList)
    return result  






## if n=3, will take k from cmd, else, from heuristic
K=-1

try: 
    n = len(sys.argv)
    goal= sys.argv[n-2]
    path= sys.argv[n-1]
    if n == 4:
        K = float(sys.argv[0])


    #file1 = pd.read_csv(path_1, header=None)
    #file1 = np.fromstring(path_1)
    
    data= np.loadtxt(path , delimiter=",")
    data = data.tolist()


    #file1 = pd.read_csv(path_1, header=None)
    #vectors = file1.to_numpy(copy=True)
    dim = len(data[0])
    if goal=="spk":
        gl= km.gl(data) ### do we need this or jacobi in c make this?
        jacobi = km.jac(gl) ###we want the points(vectors not the eigensvalues)
        eig_val= np.array(jacobi[0])
        eig_vec = np.array(jacobi[1])  # make sure that its a 2D numpy array (not an numpy array of lists)
        len = eig_val.ndim
        idx = np.argsort(eig_val)
        eig_val = eig_val[idx]
        eigvecs = eig_vec[ : , idx]    #to reordintate the vectors
        eigengap = np.diff(eig_val)
        max_gap_idx = np.argmax(eigengap[ : (np.array(eigengap).size + 1)//2])
        if K== -1:
            K = max_gap_idx + 1   
        eigenvals_k = eig_val[:K]
        eigvecs_k = eigvecs[:, :K]   #to slice the matrix 
        
        #kmeans with th k 
        resArr = initCentros(eigvecs_k,K)
        vectorslist = resArr[0]
        centroidslist = resArr[1]
        index_arrlist = resArr[2]
        epsilon =0
        iter=300
        finalCentroids = km.spk(vectorslist, centroidslist, iter, epsilon) 
        #we have to check the index (i had deleted the last check in hw2)
        if np.array(finalCentroids).size == 0:
            print("An Error Has Occurred")
            sys.exit()
        ##%0.4 for print
        PrintCentroids=[[0 for i in range(K)] for j in range(K)]

        for i in range(K):
            for j in range(K):
                PrintCentroids[i][j] = str("%.4f" % finalCentroids[i][j])
        ##print
        for i in range(np.array(index_arrlist).size):
            if i != (np.array(index_arrlist).size) - 1:
                print(str(int(index_arrlist[i])) + ",", end='')
            else:
                print(str(int(index_arrlist[i])))
        for i in range(K):  
            for j in range(K):
                if j != K - 1:
                    print(PrintCentroids[i][j] + ",", end='')
                else:
                    print(PrintCentroids[i][j])






## to know how to send to c
# send the name of func? 
    if goal == "wam":
        final = km.wam(data) 

    if goal == "ddg":
        final = km.ddg(data)

    if goal == "gl":
        final = km.gl(data)

    if goal == "wam" or goal == "ddg" or goal == "gl":
        n = len(final)
        if (len(final) ==0) :
            print("An Error Has Occurred")
            sys.exit()

        ##%0.4 for print
        PrintMatrix=[[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                PrintMatrix[i][j] = str("%.4f" % final[i][j])

        ##print
        for i in range(n):  
            for j in range(n):
                if j != n - 1:
                    print(PrintMatrix[i][j] + ",", end='')
                else:
                    print(PrintMatrix[i][j])

    if goal == "jacobi":
        final = km.jac(data)
        eigVal = final[0]
        eigVec= final[1]
        num_of_vec = len(eigVec)
        len_of_vec = len (eigVec[0])
        if (len(final) ==0) :
            print("An Error Has Occurred")
            sys.exit()

        ##%0.4 for print
        PrintVectors=[[0 for i in range(len_of_vec)] for j in range(num_of_vec)]
        for i in range(num_of_vec):
            for j in range(len_of_vec):
                PrintVectors[i][j] = str("%.4f" % eigVec[i][j])

        ##print
        for i in range(len(eigVal)):
            if i != (len(eigVal)) - 1:
                print(str("%.4f" % eigVal[i]) + ",", end='')
            else:
                print(str("%.4f" % eigVal[i]))

        for i in range(num_of_vec):  
            for j in range(len_of_vec):
                if j != len_of_vec - 1:
                    print(PrintVectors[i][j] + ",", end='')
                else:
                    print(PrintVectors[i][j])
    
   

except:
    print("An Error Has Occurred")
    sys.exit()