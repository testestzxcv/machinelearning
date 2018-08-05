from scipy import sparse

import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

from scipy import sparse

eye = np.eye(4)
print("NumPy 배열:\n{}".format(eye))

sparse_matrix = sparse.csr_matrix(eye)
print("SciPy의 CSR 행렬:\n{}".format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n{}".format(eye_coo))

# %matplotlib inline
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.sin(x)

plt.plot(x, y, marker="x")
# plt.show()

from IPython.display import display
import pandas as pd

data = {'Name':["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)
display(data_pandas)

display(data_pandas[data_pandas.Age > 30])