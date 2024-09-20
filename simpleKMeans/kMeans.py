import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

k = 3  
saveImg = True 

# Cluster class definition
class Cluster:
    def __init__(self):
        self.mean = None
        self.members = []  
        self.prevMembers = []  
        self.color = next(colors)  

   
    def setPrevMembers(self):
        self.prevMembers = self.members.copy()

  
    def addMember(self, pt):
        self.members.append(pt)

  
    def isChanged(self):
        return self.members != self.prevMembers


    def getMean(self):
        if len(self.members) == 0:
            self.mean = [-999, -999]  
        else:
            x, y = zip(*self.members)
            self.mean = [np.mean(x), np.mean(y)]


    def getTotalSquareDistance(self):
        return sum(computeDistance(self, p)**2 for p in self.members)


def computeDistance(cluster, pt):
    return np.sqrt((pt[0] - cluster.mean[0])**2 + (pt[1] - cluster.mean[1])**2)


def classify(clusters, pt):
    return min(range(k), key=lambda i: computeDistance(clusters[i], pt))


def plotClusters(clusters, i):
    plt.clf()
    for c in clusters:
        if c.members:
            x, y = zip(*c.members)
            mx, my = c.mean
            plt.scatter(x, y, color=c.color, label=f'Cluster {i}')
            plt.scatter(mx, my, color='black', marker='x', s=100)
    plt.title(f'Iteration {i}')
    if saveImg:
        plt.savefig(f'kmeans_{i}_Clusters.png')


np.random.seed(9)
data = []
for i in range(k):
    for _ in range(100):
        data.append([np.random.normal(4*i), np.random.normal(i)])


x1, y1 = zip(*data)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x1, y1)
plt.savefig('kmeansStep_Init.png')


findKPoints = []
k = 1  


while k <= 9:
 
    colors = iter(cm.rainbow(np.linspace(0, 1, k)))
    clusters = [Cluster() for _ in range(k)]
    for c in clusters:
        c.mean = random.choice(data)


    for p in data:
        clusterNum = classify(clusters, p)
        clusters[clusterNum].addMember(p)


    ischange = True
    i = 1
    while ischange and i < 1000:
        ischange = False

        for c in clusters:
            c.getMean()
            c.setPrevMembers()
            c.members = []


        for p in data:
            clusterNum = classify(clusters, p)
            clusters[clusterNum].addMember(p)

        for c in clusters:
            if c.isChanged():
                ischange = True

        i += 1


    plotClusters(clusters, k)

 
    totalSquareDistance = sum(c.getTotalSquareDistance() for c in clusters)
    findKPoints.append([k, totalSquareDistance])
    k += 1  

x, y = zip(*findKPoints)
plt.clf()
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.plot(x, y, marker='o')
plt.title('Elbow Curve to Find Optimal k')
plt.savefig('kmeans_findK.png')
