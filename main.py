import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import random

with open('images.data', 'rb') as f:
    data = pickle.load(f)

orig_data = np.array(data)

imgs = np.reshape(orig_data, (100,128*128,3))
averages = np.mean(imgs, axis=1)
print(averages.shape)

clusters2 = [2,4,6,8]

for c2 in clusters2:
    kmeans = KMeans(n_clusters=c2, random_state=0).fit(averages)
    labels = kmeans.labels_
    stratified_data = []
    for i in range(c2):
        lst = []
        for j,val in enumerate(labels):
            if i == val:
                lst.append(orig_data[j])
        stratified_data.append(lst)
    for batch in stratified_data:


#clusters = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
clusters = [64]

error = []
for c in clusters:
    print(c)
    comp_data = orig_data.copy()
    lst = []
    img = np.reshape(comp_data[0], (128*128,3))
    tot = img
    lst.append(img)
    for i in range(1, comp_data.shape[0]):
        img = np.reshape(comp_data[i], (128*128,3))
        tot = np.concatenate((tot,img))
        lst.append(img)

    kmeans = KMeans(n_clusters=c, random_state=0).fit(tot)
    labels = kmeans.labels_
    centers = np.around(kmeans.cluster_centers_)

    for i in range(len(lst)):
        for j in range(128*128):
            lst[i][j] = centers[labels[(i*(128*128))+j]]

    # Error measurement for pics
    mse = (np.square(orig_data - comp_data)).mean(axis=None)
    error.append(mse)

# plt.plot(clusters, error)
# plt.show()
# quit()

samp = random.sample(range(orig_data.shape[0]), 3)

pic1 = np.reshape(orig_data[samp[0]], (128,128,3))
pic2 = np.reshape(comp_data[samp[0]], (128,128,3))
plt.imshow(pic1)
plt.show()
plt.imshow(pic2)
plt.show()

pic1 = np.reshape(orig_data[samp[1]], (128,128,3))
pic2 = np.reshape(comp_data[samp[1]], (128,128,3))
plt.imshow(pic1)
plt.show()
plt.imshow(pic2)
plt.show()

pic1 = np.reshape(orig_data[samp[2]], (128,128,3))
pic2 = np.reshape(comp_data[samp[2]], (128,128,3))
plt.imshow(pic1)
plt.show()
plt.imshow(pic2)
plt.sho