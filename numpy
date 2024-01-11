import numpy as np

# finding even numbers using where condition
a = [11,17,12,19,26]
idx = np.array(a)
print("arrays :",idx)
res = np.where(idx % 2 == 0)
print("even numbers: ",idx[res])

# finding odd numbers using where condition
x = [12,14,8,9]
idx1 =  np.array(x)
print("arrays :", idx1)
y = np.where(idx1 % 2 != 0)
print("Odd numbers: ", idx1[y])

num = [22,12,10,2]
num1 = np.array(num)
print("arrays : ", num1)
even = np.where(num1 % 2 == 0)
print("even numbers : ", num1[even])

num2 = [23,15,9,7]
n = np.array(num2)
print("arrays : ", n)
n1 = np.where(n % 2 == 0)
print("even numbers : ", n[n1])

# finding shape of array
A = [[1,2,3],[4,5,6]]
A1 = np.array(A)
print(A1)
b = np.shape(A1)
print(b)
print(A1[0:2 ,1])
print(A1.ndim)  # finding dimension of array

a = [[1,2,3],[4,5,6],[7,8,9]]
a1 = np.array(a)
print(a1)
b1 = np.shape(a1)
print(b1)
print(a1[2, 2])
print(a1.ndim)  

arr = [[1,0,3],[0,5,0],[7,0,9]]
arr1 = np.array(arr)
print(arr1)

# finding greater then "0" elements in 2D array
arr2 = np.where(arr1 > 0)
print("elements are above 0 : ", arr1[arr2])

# Replacing value "0" with "100"
arr1[arr1 == 0] = 100
print(arr1)

arr = [[1,0,3],[0,5,0],[7,0,9]]
arr = np.array(arr)
print(arr)
arr[arr == 0] = 100
print(arr)

# Image1 with shape 3*3 and part image 2 3*6. Create a fulll image
a = [[1,2,3],[4,5,6],[7,8,9]]
b = [[10,11,12,13,14,15],[16,17,18,19,20,21],[22,23,24,25,26,27]]
a = np.array(a)
b = np.array(b)
a1 = a.reshape(-1) # converting 2D array into 1D array
b1 = b.reshape(-1) # converting 2D array into 1D array
c = np.concatenate((a1,b1)) # concatinating two arrays
print(np.shape(c)) # Shape of final array
print(c)
print(c.reshape(3,9)) # converting 1D array into 2d array

# Create 1d array with integers 0 to 9
a =np.arange(10)
print(a)

# random 1d array elements greater then the mean of the original array
a =  np.random.randint(25, size = (5, 5))
b = np.mean(a)
a1 = []
for ele in np.nditer(a):
    if ele > b:
        a1.append(ele)
a2 = np.array(a1)
print("Original array: " ,a)
print("mean value : ", b)
print("greater then mean value : ", a2)

# Calculate and print the mean, median, minimum, and maximum values of the array.
a =  np.random.randint(25, size = (5, 5))
print("Original array :" ,a)
print("Mean_value   : ",  np.mean(a))
print("median_value : ", np.median(a))
print("max_value    : ",np.max(a))
print("min_value    : ", np.min(a))

# Perform element-wise addition, subtraction, multiplication, and division on these arrays.
a = [[1,2,3],[4,5,6],[7,8,9]]
b = [[7,8,9],[1,2,3],[4,5,6]]
a1 = np.array(a)
b1 = np.array(b)
print("array 1 : ", a1)
print("array 2 : ", b1)
print("addition : ", a1 + b1)
print("subtraction : ", a1 - b1)
print("division : ", a1 / b1)
print("Multiplication : ", a1 * b1)

# Find out element occurance count in array
b = [1,2,1,3,5,1,3]
b1 = np.array(b)
b_cnt = {}
for num in range(b1.shape[0]):
    if not b1[num] in b_cnt:
        b_cnt[b1[num]] = 1
    else:
        b_cnt[b1[num]] = b_cnt[b1[num]]+1
print(b_cnt)

# dstack examples
a = [[1,2],[3,4]]
b = [[5,6],[7,8]]
a1 = np.array(a)
b1 = np.array(b)
c = np.dstack((a,b))
print(c)

x = np.zeros((3,3), dtype = "int")
y = np.ones((3,3), dtype = "int")
z = np.dstack((x,y))
print(z)

a = np.zeros((255, 255), dtype = 'int')
print(a)
b = np.ones((255 , 255), dtype = 'int')
print(b)
c = np.zeros((255, 255), dtype = 'int')
print(c)
z = np.dstack((a,b,c))
print(z)
print(np.shape(z))

# create a NumPy array from a Python list?
list = [1,2,3,4,5]
list_1 = [[1,2,3],[4,5,6]]
arr = np.array(list)
arr_1 =np.array(list_1)
print(arr)
print(arr_1)
arr_2 = np.array([[2,3,4],[5,6,7]])
print(arr_2)

# Write code to generate an array of zeros with five elements
arr_zeros = np.zeros(5)  # default data type float
print(arr_zeros)

arr_zeros_1 = np.zeros(5, dtype = int) # data type int
print(arr_zeros_1)

# What is the result of adding two NumPy arrays of the same shape?
arr_a = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr_b =  np.array([[2,3,4],[5,6,7],[7,8,9]])
arr_add = np.add(arr_a, arr_b)
print(arr_add)
print(arr_a + arr_b)

# How can you find the length of a NumPy 1D array?
arr_1d = np.random.randint((50), size = 20)
print(arr_1d)
print(arr_1d.shape)

arr1_1d = np.array([44,55,66,7,2,4,8,9,3,5,6,7,8])
print(arr1_1d.shape)

# Create a 3x3 identity matrix using NumPy.
arr_id = np.eye(3)
print(arr_id)

arr_id_1 = np.eye((3), dtype = int)
print(arr_id_1)

# Multiply each element of a NumPy array by 2
arr_mul = np.random.randint((50),size = 10 )*2
print(arr_mul)

arr_mul_1= np.random.randint((50),size = 10 )
arr_m1 = arr_mul *2
print(arr_mul_1)
print(arr_m1)

# Find the square root of each element in a given array
arr_sq = np.random.randint((50), size = 5)
arr_sqrt = np.sqrt(arr_sq)
print(arr_sqrt)

# Concatenate two arrays of different shapes vertically
arr_a = np.random.randint((10), size = (3,3))
arr_b = np.random.randint((15), size = (4,2))
arr_stack = np.vstack((arr_a, arr_b))
print(arr_stack)

# Extract the third element from a 1D array.
arr_slice = np.random.randint((40), size = 10)
print(arr_slice)
arr_ele= arr_slice[2]
print(arr_ele)

# Determine the data type of elements in a NumPy array.
arr_type = np.random.randint((5), size = 10)
print(arr_type.dtype)

arr_type = np.array(["how", "are", "you"])
print(arr_type.dtype)

# Perform element-wise multiplication of two NumPy arrays with shapes (3, 3) and (3, 3)
arr_1 = np.random.randint((5), size = (3,3))
arr_2 = np.random.randint((5), size = (3,3))
arr_mul = np.multiply(arr_1 , arr_2)
arr_mul_1 = arr_1 * arr_2
print(arr_mul)
print(arr_mul_1)

# Write a NumPy code to calculate the sum of elements in a given array.
arr_s = np.random.randint((5),  size =(3,3))
arr_sum = np.sum(arr_s)
print(arr_s)
print(arr_sum) 

# How can you transpose a NumPy array
arr_tran = np.random.randint((5), size = (4,2))
print(arr_tran)
arr_tns = np.transpose(arr_tran)
print(arr_tns)

# diagonal array addition with number "1"
a = [[1,5,8],[9,7,9],[18,20,25]]
a = np.array(a)
a1 = []
for i in range (a.shape[0]):
    for j in range (a.shape[1]):
        if i == j :
            a1.append(a[i,j]+1)
        else:
            a1.append(a[i,j])
a1 = np.array(a1)
a2 = a1.reshape(3,3)
print(a2)

# lower_digonal sum

a = [[1,2,3],[4,5,6],[7,8,9]]
a = np.array(a)
print(a)
sum = 0
for i in range(a.shape[0]):
    for j in range (a.shape[1]):
        if i > j:
            sum = sum + a[i,j]
print(sum)

#upper_diagonal sum
b = [[1,2,3],[4,5,6],[7,8,9]]
b = np.array(b)
print(b)
sum = 0
for i in range(b.shape[0]):
    for j in range (b.shape[1]):
        if i < j:
            sum = sum + b[i,j]
print(sum)

# sum of 2d array
c = [[1,2,3],[4,5,6],[7,8,9]]
c = np.array(c)
c1 = np.sum(c)
print(c1)

# sum of row wise elements
c = [[1,2,3],[4,5,6],[7,8,9]]
c = np.array(c)
c1 = np.sum(c, axis = 0)  # axis = 0 is represents rows
print(c1)

# sum of column wise elements
c = [[1,2,3],[4,5,6],[7,8,9]]
c = np.array(c)
c1 = np.sum(c, axis = 1)  # axis = 1 is represents columns
print(c1)

# minimum value of array
c2 = [[1,2,3],[8,5,6],[9,8,9]]
c2 = np.array(c2)
c3 = np.min(c2, axis = 1)  # axis = 1 is represents columns
print(c3)

# minimum value of column wise
a2 = [[1,7,10],[8,5,6],[9,8,9]]
a2 = np.array(a2)
a3 = np.min(a2, axis = 0)  # axis = 1 is represents columns
print(a3)

# Mathematical functions axis wise and entire array
a = [[20, 10,22],[21,15,10],[18,19,25]]
a1 = np.array(a)
print(a1)
a_min = np.min(a)
a_max = np.max(a)
a_sum = np.sum(a)
a_min1 = np.min(a, axis = 0)
a_min2 = np.min(a, axis = 1)
a_max1 = np.max(a, axis = 0)
a_max2 = np.max(a, axis = 1)
a_sum1 = np.sum(a, axis = 0)
a_sum2 = np.sum(a, axis = 1)
print(a_min)
print(a_max)
print(a_sum)
print(a_min1)
print(a_min2)
print(a_max1)
print(a_max2)
print(a_sum1)
print(a_sum2)

# index values of mathmatical functions.it returns indices only
a = [[20, 10,22],[21,15,30],[5,19,16]]
a1 = np.array(a)
print(a1)
a_min = np.argmin(a1)
a_max = np.argmax(a1)
a_min1 = np.argmin(a1, axis = 0)
a_min2 = np.argmin(a1, axis = 1)
a_max1 = np.argmax(a1, axis = 0)
a_max2 = np.argmax(a1, axis = 1)
print(a_min)
print(a_max)
print("**********")
print(a_min1)
print("**********")
print(a_min2)
print("**********")
print(a_max1)
print("**********")
print(a_max2)


a = [[20, 10,22],[21,15,30],[5,19,16]]
a1 = np.array(a)
print(a1)
a2 = np.argsort(a1) # returns sort indices based on values
print(a2)
a3 = np.sort(a1, axis = 0) # returns sort values row wise
print(a3)
a4 = np.sort(a1, axis = 1)  # returns sort values column wise
print(a4)
a5 = np.argwhere(a1>10) # returns indices of greater then "10" values
print(a5)

a = [20, 10,12,13,15,4]
a1 = np.array(a)
a2 = np.bincount(a1) # count the elements 
print(a2)

# returns indices of non zero values
a = [[1,2,3],[0,4,5]]
a1 = np.array(a)
indices = np.nonzero(a1) 
print(a1[indices])

# Stacks
a = [[1,2,3],[4,5,6],[7,8,9]]
b = [[10,11,12],[13,14,15],[16,17,18]]
c = [[19,20,21],[22,23,24],[25,26,27]]
a1 = np.array(a)
b1 = np.array(b)
c1 = np.array(c)
print(a)
print(b)
a2 = np.stack((a1,b1)) # combine two arrays as back to back arrays
print("***stack***")
print(a2)
print(a2.shape)
a3 = np.hstack((a1,b1)) # combine two arrays as side by side (horizontal position)
print("***hstack***")
print(a3)
print(a3.shape)
a5 = np.vstack((a1,b1)) # combine two arrays as vertical position
print("***vstack***")
print(a5)
print(a5.shape)
a4 = np.dstack((a1,b1,c1)) # combine arrays as back to back arrays
print("***dsatck***")
print(a4)
print(a4.shape)

# Write a Python function that takes an array as input and returns a list of elements that appear more than once in the array.
a = [1, 2, 3, 1, 2, 4, 5, 6, 4]
a1 = np.array(a)
unique_values, duplicate_values = np.unique(a1, return_counts=True)
print("duplicate_values : ", duplicate_values)
a3 = unique_values[np.where(duplicate_values > 1)]
print("more then one arrays : ", a3)

# Write a code snippet to obtain the sorted unique elements from a numpy array.
# array = [3, 2, 1, 2, 3, 4, 5, 6, 4]   output:[1, 2, 3, 4, 5, 6]
arr = [3, 2, 1, 2, 3, 4, 5, 6, 4]
arr1 = np.array(arr)
print(arr1)
print(np.unique(arr1))

# Given a 2D array, apply numpy.unique to find and print unique rows.
# array_2d = np.array([[1, 2, 3],[4, 5, 6],[1, 2, 3],[7, 8, 9]]) output: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr = [[1, 2, 3],[4, 5, 6],[1, 2, 3],[7, 8, 9]]
arr1 = np.array(arr)
print(arr1)
arr2 = np.unique(arr1, axis = 0)
print(arr2)

# Create a numpy array of strings and use numpy.unique to find and print unique
# array_strings = np.array(['apple', 'banana', 'orange', 'apple', 'kiwi', 'banana'])
# output: ['apple' 'banana' 'kiwi' 'orange']
str_arr = ['apple', 'banana', 'orange', 'apple', 'kiwi', 'banana']
arr1 = np.array(str_arr)
unique_str = np.unique(arr1)
unique_str, indices = np.unique(arr1, return_index= True)
unique_str, counts = np.unique(arr1,  return_counts= True)
print(arr1)
print(indices)
print(unique_str)
duplicates = unique_str[np.where(counts>1)]
print(duplicates)

#  Find the least repeated element(s) in the array.
nums = np.random.randint(10, size =(10, ))
print(nums)
print(nums.shape)
values, count = np.unique(nums, return_counts = True)
duplicates =values[np.where(count == np.min(count))]
print(duplicates)
print(duplicates.shape)

# How is vstack() different from hstack() in NumPy

# vstack()
ar  = [[1,2,3],[4,5,6],[7,8,9]]
ar1 =[[0,9,8],[8,7,6],[4,3,2]]
ar  = np.array(ar)
ar1 = np.array(ar1)
ar2 = np.vstack((ar, ar1)) # two arrays will be form in vertical direction.
print(ar2)
print(ar2.shape)
print("***************")

# hstack()
arr  = [[1,2,3],[4,5,6],[7,8,9]]
arr1 =[[0,9,8],[8,7,6],[4,3,2]]
arr  = np.array(arr)
arr1 = np.array(arr1)
arr2 = np.hstack((arr, arr1)) # two arrays will be form in horizontal direction.
print(arr2)
print(arr2.shape)

# Input_array = [1,0,3] output_array = [[0, 1, 0, 0], [1,0,0,0], [0 0 0 1]]
# It's like 1 in input_array represented as [0 1 0 0] and 3 is represented as [0 0 0 1] etc.
arr = np.array([1,0,3])
nrows = arr.shape[0]
ncols = np.max(arr)+1
output_arr = np.zeros((nrows,ncols),dtype = "int")
for rowidx, ele in enumerate(arr):
    output_arr[rowidx,ele] = 1
print(output_arr)

print(np.max(arr))
