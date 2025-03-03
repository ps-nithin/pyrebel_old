from numba import cuda
import numpy as np

@cuda.jit
def max_example(result):
    val=5
    i=cuda.grid(1)
    if i==3:
        val=7
    cuda.atomic.max(result[i],0,val)

result=np.zeros([10,1],dtype=np.float64)
max_example[1,10](result)

print(result)
