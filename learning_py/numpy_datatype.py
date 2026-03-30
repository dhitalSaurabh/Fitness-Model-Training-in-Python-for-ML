import numpy as np
arr = np.array([1,2,3,4,5])
print("Array datatype:", arr.dtype)

arr1 = np.array(["f","f","g","h","h"])
print("Array datatype:", arr1.dtype)

arr2 =  np.array([1,2,3,4,5], dtype = 'O')

print("Array:", arr2)
print("Array datatype:", arr2.dtype)

arr3 = np.array([0.0,2.2,3.3,4.4])
# newarr = arr3.astype('i')
newarr = arr3.astype(bool)

print("Original Array:", arr3)
print("Converted Array:", newarr)
print("Original Array datatype:", arr3.dtype)
print("Converted Array datatype:", newarr.dtype)