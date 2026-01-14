import numpy as np
from Kmeans_ff import FireflyClustering

np.set_printoptions(precision=3, suppress=True)
np.random.seed(1)

# DATASET
X = np.array([
    [1,2,1,2],[1,1,2,2],[2,2,1,1],[2,1,2,1],
    [8,8,9,8],[9,8,8,9],[8,9,8,8],[9,9,9,8],
    [4,5,4,5],[5,4,5,4],[4,4,5,5],[5,5,4,4]
])

# INISIAL FIREFLY
firefly1 = np.array([[1,2,1,2],[8,8,8,8],[4,5,4,5]])
firefly2 = np.array([[2,1,2,1],[9,9,8,8],[5,4,5,4]])

# JALANKAN MODEL
model = FireflyClustering(X, firefly1, firefly2)

sse1_i1, sse2_i1 = model.iterasi_1()
model.update_firefly(sse1_i1, sse2_i1)
sse1_i2, sse2_i2 = model.iterasi_2()
model.kesimpulan(sse1_i2, sse2_i2)
