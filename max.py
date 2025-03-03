from numba import cuda
import numpy as np

@cuda.jit
def find_max(arr,max_):
    i=cuda.grid(1)
    if arr[i]>max_[0]:
        max_[0]=arr[i]

max_=np.zeros(1,dtype=np.int)
max_d=cuda.to_device(max_)
arr_=np.arange(15)
arr_d=cuda.to_device(arr_)
find_max[1,15](arr_d,max_d)
cuda.synchronize()
print(max_d.copy_to_host()[0])

