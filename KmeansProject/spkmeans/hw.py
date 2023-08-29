import numpy as np

def LU_pivot(A, pivot=0):
    n = A.shape[0]
    U = np.copy(A)  
    L = np.eye(n)   # initialize L as an identity matrix
    P = np.eye(n)   
    Q = np.eye(n)   
    for i in range(n-1):
        if pivot == 1:
            max_index = np.argmax(abs(U[i:n, i])) + i
            if max_index != i:
                U[[i, max_index], :] = U[[max_index, i], :]
                P[[i, max_index], :] = P[[max_index, i], :]
        
        elif pivot == 2:
            max_index = np.argmax(abs(U[i:n, i:n]))  
            row_index = (max_index // (n-i)) + i     
            col_index = (max_index % (n-i)) + i      
            if row_index != i:
                U[[i, row_index], :] = U[[row_index, i], :]
                P[[i, row_index], :] = P[[row_index, i], :]
                Q[:, [i, row_index]] = Q[:, [row_index, i]]
            if col_index != i:
                U[:, [i, col_index]] = U[:, [col_index, i]]
                L[:, [i, col_index]] = L[:, [col_index, i]]
        
        U = U.astype(float)
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]
            L[j, i] = factor
    
    Q = Q @ P.T 
    
    return [Q, P, U, L]