# %%
from sklearn.decomposition import PCA
import umap
import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
# %%
data_path = './training_data'
data_mat = scio.loadmat(
    (os.path.join(data_path, "data64X64(75-125mm)(393-503K)ln1+gs1.mat")))
input_data_raw = data_mat['input']
input_data = input_data_raw
out_data = data_mat['output']

out_norm = (out_data/127.5)-1.0
out_norm = out_norm.reshape(out_norm.shape[0], -1)

# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(out_norm)
embedding.shape
# %%
fig = plt.figure(figsize=(4.8, 3.6))
sc = plt.scatter(
    embedding[:, 0],
    embedding[:, 1], s=5, c=input_data[:, 3], cmap='viridis')
plt.colorbar(sc)
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
# %%

pca = PCA(n_components=2)  # Choose the number of components
principal_components = pca.fit_transform(out_norm)

# %%
fig = plt.figure(figsize=(4.8, 3.6))
plt.scatter(
    principal_components[:, 0],
    principal_components[:, 1],
    s=5)
# %%
