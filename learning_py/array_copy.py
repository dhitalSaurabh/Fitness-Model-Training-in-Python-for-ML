import numpy as np
arr = np.array([1,2,3,4,5])
x = arr.copy()
y = arr.view()

arr[0] = 40

print(x.base)
print(y.base)
print("Original Array:", arr)
print("Copied Array:", x)
print("array shape", arr.shape)