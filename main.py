import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import random

with open('images.data', 'rb') as f:
    data = pickle.load(f)

orig_data = np.array(data) # retrieve image data

imgs = np.reshape(orig_data, (100,128*128,3)) # reshape data to get RGB channels
averages = np.mean(imgs, axis=1) # average along each images RGB

clusters = [16, 32, 64, 128, 256] # clusters for the image batchs
clusters2 = [2, 4, 6, 8] # clusters for average pixel values

error_per_avg_cluster = [] # stores error of each cluster value for each cluster2 value
for c2 in clusters2:
    kmeans = KMeans(n_clusters=c2, random_state=0).fit(averages) # first clustering
    labels = kmeans.labels_
    stratified_data = []

    for i in range(c2): # stratify images based on intial clustering
        lst = []
        for j,val in enumerate(labels):
            if i == val:
                lst.append(orig_data[j])
        lst = np.array(lst)
        stratified_data.append(lst)

    all_error = [] # stores all errors for each cluster value for each batch
    for batch in stratified_data:
        error = [] # stores error for this specific batch
        for c in clusters: # for each cluster value
            comp_data = batch.copy() # copy the batch and reshape 
            lst = []
            img = np.reshape(comp_data[0], (128*128,3))
            tot = img
            lst.append(img)
            for i in range(1, comp_data.shape[0]): # group all pixels of every image in the batch
                img = np.reshape(comp_data[i], (128*128,3))
                tot = np.concatenate((tot,img))
                lst.append(img)

            # cluster the pixel values
            kmeans = KMeans(n_clusters=c, random_state=0).fit(tot)
            labels = kmeans.labels_
            centers = np.around(kmeans.cluster_centers_)

            # replace pixel values with averages
            for i in range(len(lst)):
                for j in range(128*128):
                    lst[i][j] = centers[labels[(i*(128*128))+j]]

            # Error measurement for pics
            mse = (np.square(batch - comp_data)).mean(axis=None)
            error.append(mse)

        all_error.append(error) # append errors to all_error

    # average all errors across the batches to get average error of entire dataset
    all_error = np.array(all_error)
    avg_error = np.mean(all_error, axis=0)
    error_per_avg_cluster.append(avg_error) # add error for this cluster2 value, go to next one

# plot all the errors
for avg_err in error_per_avg_cluster:
    plt.plot(clusters, avg_err)
    plt.show()

# samp = random.sample(range(orig_data.shape[0]), 3)

# pic1 = np.reshape(orig_data[samp[0]], (128,128,3))
# pic2 = np.reshape(comp_data[samp[0]], (128,128,3))
# plt.imshow(pic1)
# plt.show()
# plt.imshow(pic2)
# plt.show()

# pic1 = np.reshape(orig_data[samp[1]], (128,128,3))
# pic2 = np.reshape(comp_data[samp[1]], (128,128,3))
# plt.imshow(pic1)
# plt.show()
# plt.imshow(pic2)
# plt.show()

# pic1 = np.reshape(orig_data[samp[2]], (128,128,3))
# pic2 = np.reshape(comp_data[samp[2]], (128,128,3))
# plt.imshow(pic1)
# plt.show()
# plt.imshow(pic2)
# plt.show()