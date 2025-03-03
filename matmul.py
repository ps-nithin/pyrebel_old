from numba import cuda, float32
import numpy as np
import math
import time

@cuda.jit
def matmul(A,B,C):
    i,j=cuda.grid(2)
    if i<C.shape[0] and j<C.shape[1]:
        tmp=0.0
        for k in range(A.shape[1]):
            tmp+=A[i,k]*B[k,j]
        C[i,j]=tmp

def compute_matmul(a,b):
    start_time=time.time()
    x_h=np.arange(a*b).reshape([a,b])
    y_h=np.ones([a,b])
    z_h=np.zeros([a,b])

    x_d=cuda.to_device(x_h)
    y_d=cuda.to_device(y_h)
    z_d=cuda.to_device(z_h)

    matmul[32,512](x_d,y_d,z_d)
    z_h=z_d.copy_to_host()
    print("Finished in",time.time()-start_time,"sec.")
