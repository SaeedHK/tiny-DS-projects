import pandas as pd
import numpy as np
import scipy.sparse as sps



# This reading Data may takes several minutes

df = pd.read_csv('foodsRecoded.csv' , header= None , delimiter =':' )
df_mat = df.pivot_table(index = 0 , columns= 1 , values= 2)

df_mat.index.name = 'user'
df_mat.columns.name = 'product'
print('data is read as a dataframe')

M = sps.coo_matrix(df_mat.fillna(0))
print('data is read as a sprase matrix')

I , J , VAL = sps.find(M)

m = M.shape[0]
n = M.shape[1]

def MF ( I , J , VAL , m , n , k=20 , n_steps = 80 , eps = 0.1 , tresh=0.1):    

    U = np.random.random([m,k])
    V = np.random.random([k,n])

    #E = np.ones([m,n])
    #while (E.sum().sum() > tresh):

    for t in range(n_steps):

        # print('step is ' , t)

        K = np.zeros([m,n])
        E = np.zeros([m,n])
        K_U = np.zeros([m,k])
        K_V = np.zeros([k,n])

        for u in range(len(I)):
            K[I[u],J[u]] = np.dot(U[I[u],:] , V[:,J[u]]) - VAL[u]
            E[I[u],J[u]] = K[I[u],J[u]]**2


        # K = np.matmul(U,V) - M

        for u in range(len(I)):
            K_U[I[u],:] += K[I[u],J[u]]*np.transpose(V[:,J[u]])
            K_V[:,J[u]] += K[I[u],J[u]]*np.transpose(U[I[u],:])

        # K_U = np.matmul(K,np.transpose(V)) #shape m*n * n*k = m*k
        # K_V = np.matmul(np.transpose(U),K) #shape k*m * m*n = k*n

        U = U - ((eps/(len(I))) * K_U)
        V = V - ((eps/(len(I))) * K_V)

    print(E.sum().sum()/(len(I)))
    #print(M)
    #print(np.matmul(U,V))

print(MF(I,J,VAL,m,n))
