import numpy as np
from scipy.ndimage import label


# The way to find clusters on a grid

grid = np.array([[ 1,  0, -1,  0,  1],
                 [ 1, -1, -1,  0,  0],
                 [ 0,  0,  1,  0,  0],
                 [ 1,  0,  1,  1, -1],
                 [ 1,  0,  0,  0,  0]])

structure = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])

active_cells = (grid == 1) | (grid == -1)

labeled_array, num_clusters = label(active_cells, structure=structure)

cluster_filters = [(labeled_array == cluster_id) for cluster_id in range(1, num_clusters + 1)]

print(f"Number of clusters: {num_clusters}")
for i, cluster_filter in enumerate(cluster_filters, start=1):
    print(f"Cluster {i} filter:\n{cluster_filter}")
