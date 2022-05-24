import random
import matplotlib.pyplot as plt
import pickle

data = []

samp = random.sample(range(1,1201), 25)
for x in samp:
    image = plt.imread("Linnaeus_5/train/berry/%d_128.jpg"%(x))
    pic = image.reshape(128*128*3)
    data.append(pic)

samp = random.sample(range(1,1201), 25)
for x in samp:
    image = plt.imread("Linnaeus_5/train/bird/%d_128.jpg"%(x))
    pic = image.reshape(128*128*3)
    data.append(pic)

samp = random.sample(range(1,1201), 25)
for x in samp:
    image = plt.imread("Linnaeus_5/train/dog/%d_128.jpg"%(x))
    pic = image.reshape(128*128*3)
    data.append(pic)

samp = random.sample(range(1,1201), 25)
for x in samp:
    image = plt.imread("Linnaeus_5/train/flower/%d_128.jpg"%(x))
    pic = image.reshape(128*128*3)
    data.append(pic)

with open('images.data', 'wb') as f:
    pickle.dump(data, f)