# Numpy-Cheatsheet for Developers

## Introduction-What-is-Numpy?

> NumPy (Numerical Python) is the fundamental library for numerical computing in Python, providing powerful capabilities for working with large, multi-dimensional arrays and matrices. At its core, NumPy introduces the ndarray object, which is significantly more efficient for storing and manipulating numerical data than standard Python lists, especially for large datasets. It offers a vast collection of high-level mathematical functions to operate on these arrays, covering linear algebra, Fourier transforms, random number generation, and more. NumPy's efficiency stems from its implementation in C and Fortran, allowing for high-performance operations that are crucial for scientific computing, data analysis, and machine learning, making it an indispensable building block for many other Python libraries like Pandas and Scikit-learn.


## 1. Importing NumPy

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|`import numpy`|	Imports the NumPy library, allowing you to use its functions and objects by prefixing them with numpy.|
|`import numpy as np`|	Imports the NumPy library and assigns it the conventional alias np, making it quicker to call its functions (e.g., np.array()). This is the most common and recommended way.|

## 2. Array Creation

> Learn how to generate various types of NumPy arrays, from empty arrays to arrays filled with specific values or ranges.

|Command | description|
|----------|-------------|
|`np.array([1, 2, 3])`|	Creates a 1-dimensional NumPy array from a Python list or tuple.|
|`np.array([[1, 2], [3, 4]])`|	Creates a 2-dimensional NumPy array (matrix) from nested Python lists.|
|`np.zeros((rows, cols))`|	Creates an array of specified rows and cols (or other shape) filled with zeros.|
|`np.ones((rows, cols))`|	Creates an array of specified rows and cols (or other shape) filled with ones.|
|`np.empty((rows, cols))`|	Creates an array with uninitialized (arbitrary) data of the given shape. Faster than zeros or ones.|
|`np.arange(start, stop, step)`|	Creates an array with evenly spaced values within a given interval (similar to Python's range()).|
|`np.linspace(start, stop, num)`|	Creates an array with num evenly spaced values over a specified interval [start, stop].|
|`np.full((rows, cols), value)`|	Creates an array of specified shape filled with a value.|
|`np.eye(N)`|	Creates an N x N identity matrix (a square matrix with ones on the main diagonal and zeros elsewhere).|
|`np.diag(v)`|	Creates a 2-D array with v on the diagonal and zeros elsewhere. If v is 2-D, returns the diagonal.|
|`np.fromfunction(function, shape)`|	Constructs an array by executing a function over each coordinate.|
|`np.asarray(a)`|	Converts the input a to a NumPy array. If a is already an array, no copy is made.|
|`np.fromiter(iterable, dtype, count)`|	Creates a new 1-dimensional array from an iterable object.|

## 3. Array Inspection

> Discover commands to examine an array's dimensions, data type, number of elements, and other fundamental properties.

|Command | description|
|----------|-------------|
|`arr.shape`|	Returns a tuple indicating the dimensions (shape) of the array.|
|`arr.ndim`|	Returns the number of dimensions (axes) of the array.|
|`arr.size`|	Returns the total number of elements in the array.|
|`arr.dtype`|	Returns the data type of the elements in the array.|
|`arr.itemsize`|	Returns the size in bytes of each element of the array.|
|`arr.nbytes`|	Returns the total bytes consumed by the elements of the array.|
|`type(arr)`|	Checks the Python type of the array object itself (will be numpy.ndarray).|
|`np.info(arr)`|	Provides detailed information about the array, including its memory layout, data type, and shape.
|`arr.T`|	Returns the transposed array.|
|`arr.flags`|	Returns information about the memory layout of the array (e.g., C-contiguous, Fortran-contiguous).|
|`arr.real`|	Returns the real part of the array elements.|
|`arr.imag`|	Returns the imaginary part of the array elements.|


## 4. Array Manipulation (Reshaping, Joining, Splitting)

> Understand how to change an array's shape, combine multiple arrays, or divide a single array into several smaller ones.

|Command | description|
|----------|-------------|
|`arr.reshape(shape)`|	Gives a new shape to an array without changing its data. The new shape must have the same total number of elements.|
|`arr.ravel()`|	Returns a flattened 1D array. A view is returned if possible, otherwise a copy.|
|`arr.flatten()`|	Returns a copy of the array flattened to one dimension.|
|`np.transpose(arr)`| or arr.T	Permutes the dimensions of an array. For a 2D array, this swaps rows and columns.|
|`np.concatenate((arr1, arr2), axis=0)`|	Joins a sequence of arrays along an existing axis. axis=0 for row-wise, axis=1 for column-wise (for 2D arrays).|
|`np.vstack((arr1, arr2))`|	Stacks arrays in sequence vertically (row-wise). Equivalent to np.concatenate((arr1, arr2), axis=0).|
|`np.hstack((arr1, arr2))`|	Stacks arrays in sequence horizontally (column-wise). Equivalent to np.concatenate((arr1, arr2), axis=1).|
|`np.dstack((arr1, arr2))`|	Stacks arrays in sequence depth-wise (along the third axis).|
|`np.split(arr, indices_or_sections, axis=0)`|	Splits an array into multiple sub-arrays. indices_or_sections can be an integer (number of equal splits) or a list of indices where to split.|
|`np.vsplit(arr, indices_or_sections)`|	Splits an array into multiple sub-arrays vertically (row-wise). Equivalent to np.split(arr, ..., axis=0).|
|`np.hsplit(arr, indices_or_sections)`|	Splits an array into multiple sub-arrays horizontally (column-wise). Equivalent to np.split(arr, ..., axis=1).|
|`np.append(arr, values, axis=None)`|	Appends values to the end of an array. If axis is specified, values are appended along that axis. Note: Returns a new array.|
|`np.delete(arr, obj, axis=None)`|	Returns a new array with sub-arrays along an axis deleted. obj can be a slice, an int, or array of ints.|
|`np.insert(arr, obj, values, axis=None)`|	Inserts values along the given axis before the given indices. Returns a new array.|
|`np.resize(arr, new_shape)`|	Returns a new array with the specified shape. If the new array is larger, the old array's content is repeated. If smaller, content is truncated.|
|`arr.resize(new_shape)`|	Resizes the array in-place. This method should not be used with arr as a view.|

## 5. Array Indexing and Slicing

> Explore powerful ways to access individual elements, specific rows, columns, or sub-arrays using integer, boolean, or advanced indexing.

|Command | description|
|----------|-------------|
|`arr[0]`|	Accesses the element at index 0 (first element) of a 1D array arr.|
|`arr[1, 2]`|	Accesses the element at row 1, column 2 of a 2D array arr.|
|`arr[0:5]`|	Slices the array arr from index 0 up to (but not including) index 5.|
|`arr[2:]`|	Slices the array arr from index 2 to the end.|
|`arr[:3]`|	Slices the array arr from the beginning up to (but not including) index 3.|
|`arr[::2]`|	Slices the array arr with a step of 2 (every other element).|
|`arr[::-1]`|	Reverses the array arr.|
|`arr[:, 1]`|	Slices all rows and selects the column at index 1 of a 2D array.|
|`arr[0, :]`|	Slices the first row and selects all columns of a 2D array.|
|`arr[1:3, 0:2]`|	Selects rows from index 1 to 2, and columns from index 0 to 1 of a 2D array.|
|`arr[[0, 2, 4]]`|	Fancy Indexing: Selects elements at specific, non-contiguous indices (0, 2, and 4) from a 1D array.|
|`arr[[0, 1], [2, 3]]`|	Fancy Indexing: Selects elements (arr[0,2], arr[1,3]) from a 2D array.|
|`arr[arr > 5]`|	Boolean Indexing: Returns a new array containing only elements from arr that are greater than 5.|
|`arr[(arr > 5) & (arr < 10)]`|	Boolean Indexing: Returns elements satisfying multiple conditions.|
|`arr[np.newaxis, :]`|	Expands the array arr by adding a new axis (e.g., turns 1D into 2D as a row vector).|
|`arr.reshape(rows, cols)`|	Reshapes the array arr to a new shape (e.g., from 1D to 2D).|
|`arr.flatten()`|	Returns a copy of the array collapsed into one dimension.
|`arr.ravel()`|	Returns a flattened view of the array (if possible).|

## 6. Mathematical Operations (Element-wise)

> This covers applying arithmetic, trigonometric, and other functions to each element of an array individually.

|Command | description|
|----------|-------------|
|`np.add(arr1, arr2) or arr1 + arr2`|	Performs element-wise addition between two arrays or an array and a scalar.|
|`np.subtract(arr1, arr2) or arr1 - arr2`|	Performs element-wise subtraction between two arrays or an array and a scalar.|
|`np.multiply(arr1, arr2) or arr1 * arr2`|	Performs element-wise multiplication between two arrays or an array and a scalar.|
|`np.divide(arr1, arr2) or arr1 / arr2`|	Performs element-wise division between two arrays or an array and a scalar.|
|`np.power(arr, exponent) or arr **`| exponent	Raises each element of an array to the power of a specified exponent (element-wise).|
|`np.sqrt(arr)`|	Calculates the element-wise square root of an array.|
|`np.exp(arr)`|	Calculates the element-wise exponential (e^x) of an array.|
|`np.log(arr)`|	Calculates the element-wise natural logarithm (lnx) of an array.|
|`np.log10(arr)`|	Calculates the element-wise base-10 logarithm (log 10x) of an array.|
|`np.sin(arr)`|	Calculates the element-wise sine of an array (in radians).|
|`np.cos(arr)`|	Calculates the element-wise cosine of an array (in radians).|
|`np.abs(arr)`| or np.absolute(arr)	Calculates the element-wise absolute value of an array.|
|`np.ceil(arr)`|	Calculates the element-wise ceiling of an array (rounds up to the nearest integer).|
|`np.floor(arr)`|	Calculates the element-wise floor of an array (rounds down to the nearest integer).|
|`np.round(arr, decimals=0)`|	Rounds each element to the given number of decimals.|
|`np.maximum(arr1, arr2)`|	Compares two arrays element-wise and returns the maximum value for each position.|
|`np.minimum(arr1, arr2)`|	Compares two arrays element-wise and returns the minimum value for each position.|
|`np.mod(arr1, arr2) or arr1 % arr2`|	Performs element-wise modulo (remainder after division).|
|`np.greater(arr1, arr2) or arr1 > arr2`|	Performs element-wise comparison, returning True if arr1 element is greater than arr2 element.|
|`np.less(arr1, arr2) or arr1 < arr2`|	Performs element-wise comparison, returning True if arr1 element is less than arr2 element.|
|`np.equal(arr1, arr2) or arr1 == arr2`|	Performs element-wise comparison, returning True if arr1 element is equal to arr2 element.|

## 7. Mathematical Operations (Aggregate)

> Learn how to perform summary calculations across an entire array or along specific axes, such as sums, means, and standard deviations.

|Command | description|
|----------|-------------|
|`np.sum(arr)`|	Calculates the sum of all elements in an array.|
|`np.sum(arr, axis=0)`|	Calculates the sum along a specific axis (e.g., columns).|
|`np.sum(arr, axis=1)`|	Calculates the sum along a specific axis (e.g., rows).|
|`np.mean(arr)`|	Calculates the arithmetic mean (average) of all elements.|
|`np.mean(arr, axis=0)`|	Calculates the mean along a specific axis.|
|`np.std(arr)`|	Calculates the standard deviation of all elements.|
|`np.std(arr, axis=1)`|	Calculates the standard deviation along a specific axis.|
|`np.min(arr) or arr.min()`|	Finds the minimum value in the array.|
|`np.max(arr) or arr.max()`|	Finds the maximum value in the array.|
|`np.argmin(arr) or arr.argmin()`|	Returns the index of the minimum value.|
|`np.argmax(arr) or arr.argmax()`|	Returns the index of the maximum value.|
|`np.median(arr)`|	Calculates the median of the array elements.|
|`np.percentile(arr, q)`|	Computes the q-th percentile of the data.|
|`np.var(arr)`|	Calculates the variance of the array elements.|
|`np.prod(arr)`|	Calculates the product of all elements in an array.|
|`np.cumsum(arr)`|	Returns the cumulative sum along a given axis.|
|`np.cumprod(arr)`|	Returns the cumulative product along a given axis.|
|`np.any(arr)`|	Tests whether any array element along a given axis evaluates to True.|
|`np.all(arr)`|	Tests whether all array elements along a given axis evaluate to True.|

## 8. Broadcasting

> Grasp the fundamental concept that allows NumPy to perform operations on arrays of different shapes automatically.

|Command | description|
|----------|-------------|
|`arr + scalar`|	Adds a scalar value to every element in an array.|
|`arr - scalar`|	Subtracts a scalar value from every element in an array.|
|`arr * scalar`|	Multiplies every element in an array by a scalar value.|
|`arr / scalar`|	Divides every element in an array by a scalar value.|
|`arr1 + arr2`|	Adds two arrays element-wise, if their shapes are compatible for broadcasting.|
|`arr1 * arr2`|	Multiplies two arrays element-wise, if their shapes are compatible for broadcasting.|
|`np.array([[1], [2], [3]]) + np.array([10, 20, 30])`|	Example of adding a 2D column array to a 1D row array, broadcasting both.|
|`np.arange(12).reshape(3,4) + np.array([1,2,3,4])`|	Example: adding a 1D array (row vector) to each row of a 2D array.|
|`np.arange(3).reshape(3,1) + np.arange(3)`|	Example: adding a 1D array (row vector) to a 2D column vector, resulting in a 2D array.|


## 9. Linear Algebra

> Access specialized functions for common linear algebra tasks like dot products, matrix multiplication, and solving systems of equations.

|Command | description|
|----------|-------------|
|`np.dot(a, b)`|	Computes the dot product of two arrays a and b. For 2-D arrays, it's matrix multiplication.|
|`np.matmul(a, b)`|	Performs matrix product of two arrays. Equivalent to @ operator for 2-D arrays.|
|`np.linalg.inv(a)`|	Computes the (multiplicative) inverse of a square matrix a.|
|`np.linalg.det(a)`|	Computes the determinant of an array a.|
|`np.linalg.eig(a)`|	Computes the eigenvalues and right eigenvectors of a square array.|
|`np.linalg.eigvalsh(a)`|	Computes the eigenvalues of a Hermitian or real symmetric matrix.|
|`np.linalg.svd(a)`|	Performs Singular Value Decomposition (SVD) of a matrix a.|
|`np.linalg.solve(a, b)`|	Solves a linear matrix equation, or system of linear scalar equations. (ax = b)|
|`np.linalg.lstsq(a, b)`|	Computes the least-squares solution to a linear matrix equation.|
|`np.linalg.qr(a)`|	Computes the QR factorization of a matrix.|
|`np.linalg.norm(a)`|	Computes the Frobenius norm of a matrix or vector norm.|
|`np.transpose(a) or a.T`|	Permutes the dimensions of an array, often used for matrix transpose.|
|`np.trace(a)`|	Computes the sum along diagonals of a given array.|
|`np.identity(n)`|	Returns the identity array of size n x n.|
|`np.eye(N, M=None, k=0)`|	Returns a 2-D array with ones on the diagonal and zeros elsewhere.|
|`np.linalg.matrix_rank(M)`|	Returns the rank of a matrix M using SVD method.|

## 10. Random Number Generation

> This section provides tools for creating arrays filled with random numbers, useful for simulations and data generation.

|Command | description|
|----------|-------------|
|`np.random.rand(d0, d1, ...)`|	Creates an array of the given shape, filled with random samples from a uniform distribution over [0, 1).|
|`np.random.randn(d0, d1, ...)`|	Creates an array of the given shape, filled with random samples from a standard normal (Gaussian) distribution (mean 0, variance 1).|
|`np.random.randint(low, high=None, size=None, dtype=int)`|	Returns random integers from low (inclusive) to high (exclusive). size specifies the output shape.|
|`np.random.random(size=None)`|	Returns random floats in the half-open interval [0.0, 1.0). Similar to np.random.rand() but takes size as a tuple.|
|`np.random.choice(a, size=None, replace=True, p=None)`|	Generates a random sample from a given 1-D array (a). size specifies the output shape, replace allows sampling with replacement, and p provides probabilities for each element.|
|`np.random.seed(seed=None)`|	Seeds the pseudo-random number generator. Useful for reproducible results: if you use the same seed, you'll get the same "random" numbers.|
|`np.random.shuffle(x)`|	Modifies a sequence x in-place by shuffling its contents.|
|`np.random.permutation(x)`|	Randomly permutes a sequence, or returns a permuted range. Unlike shuffle, it returns a new array.|
|`np.random.normal(loc=0.0, scale=1.0, size=None)`|	Draws random samples from a normal (Gaussian) distribution. loc is the mean, scale is the standard deviation.|
|`np.random.uniform(low=0.0, high=1.0, size=None)`|	Draws samples from a uniform distribution over the half-open interval [low, high).|
|`rng = np.random.default_rng(seed=None)`|	Creates a new Generator with a default BitGenerator. This is the recommended modern way to generate random numbers in NumPy for better practice and flexibility.|
|`rng.random(size=None)`|	Generates random floats using the new Generator object. (Example usage with rng).|
|`rng.integers(low, high=None, size=None, endpoint=False)`|	Generates random integers using the new Generator object. endpoint=True makes high inclusive.|

## 11. Saving and Loading Arrays

> Learn how to store your NumPy arrays to disk and load them back into memory for later use.

|Command | description|
|----------|-------------|
|`np.save('filename.npy', array)`|	Saves a single NumPy array to a binary .npy file.|
|`np.load('filename.npy')`|	Loads a single NumPy array from a binary .npy file.|
|`np.savez('archive.npz', array1=arr1, array2=arr2)`|	Saves multiple NumPy arrays into a single uncompressed .npz archive. Arrays are saved as keyword arguments.|
|`np.load('archive.npz')`|	Loads data from an .npz archive. Returns a NpzFile object, which is a dictionary-like object where arrays can be accessed by their keyword names (e.g., data['array1']).|
|`np.savez_compressed('archive.npz', array1=arr1, array2=arr2)`|	Saves multiple NumPy arrays into a single compressed .npz archive. This is useful for large arrays to save disk space.|
|`np.loadtxt('data.txt')`|	Loads data from a text file (e.g., .txt, .csv) into a NumPy array. Assumes numeric data and common delimiters.|
|`np.savetxt('output.txt', array, fmt='%.2f', delimiter=',')`|	Saves a NumPy array to a text file. fmt specifies the format of numbers (e.g., 2 decimal places), delimiter specifies the column separator.|
|`np.genfromtxt('data.csv', delimiter=',')`|	Loads data from a text file, similar to loadtxt, but handles missing values more gracefully.|

## 12. Boolean Indexing and Masking

> Understand how to select and modify array elements based on conditions, creating powerful filters for your data.

|Command | description|
|----------|-------------|
|`arr[arr > 5]`|	Selects elements from arr where the condition arr > 5 is True.|
|`arr[arr % 2 == 0]`|	Selects elements from arr that are even.|
|`(arr > 0) & (arr < 10)`|	Combines multiple boolean conditions using & (AND). Parentheses are crucial.|
|`arr[~condition]`|	Selects elements where the condition is False (inverts the boolean mask).|
|`np.where(condition, x, y)`|	Returns elements chosen from x or y depending on condition. x is returned where condition is True, y where False.|
|`arr[mask]`|	Applies a boolean array mask (of the same shape) to arr to select elements.|
|`arr[[True, False, True]]`|	Direct boolean indexing using a list or array of booleans.|
|`arr[arr_2d[:, 0] > 0]`|	Boolean indexing on a 2D array, selecting rows based on a condition on one column.|
|`arr[condition] = value`|	Assigns a value to all elements of arr where condition is True.|
|`arr[condition] = other_arr[condition]`|	Assigns values from other_arr to arr where condition is True.|












