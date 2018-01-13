import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target

class RIM:
    def __init__(self,X, y, K,alfa):
        self.K = K
        self.data = np.insert(X, 0, y, axis=1)
        self.data = np.random.permutation(self.data)
        self.X = self.data[:,1:]
        self.y = self.data[:,0]

        self.cluster_by_k_means = defaultdict(list)
        self.alfa = alfa
        self.normalize()
        self.X = np.insert(self.X,0,1,axis=1)
        print(self.X.shape)
        self.lambdas = np.random.rand(self.X.shape[0], self.X.shape[1])
        print(self.lambdas)
       # print(len(self.X))

    def normalize(self):
        means = np.mean(self.X, axis = 0)
        stds = np.std(self.X, axis = 0)
        self.X = (self.X - means)/stds

    def initialize(self):
        self.kmeans = KMeans(n_clusters=self.K,init="k-means++").fit(self.X)
        for i in range(len(self.X)):
            self.cluster_by_k_means[self.kmeans.labels_[i]].append(self.X[i])
        print(len(self.cluster_by_k_means[0]))

    def test(self):
        print(self.y)
        print(self.kmeans.labels_)
        acc = accuracy_score(self.y, self.kmeans.labels_)
        mi = normalized_mutual_info_score(self.y, self.kmeans.labels_)
        print(mi)
        print(acc)

    def conditional_propability_of_y(self):
        p = []
        for j in range(len(self.X)):
            #print(np.sum(np.exp(np.dot(self.lambdas,self.X[j]))))
            p.append(np.exp(np.dot(self.lambdas[j],self.X[j]))/np.sum(np.exp(np.dot(self.lambdas,self.X[j]))))

        #print(p)


#X = np.array([[1,1,1],[2,3,4], [4,5,6]])
r = RIM(X,y,3,10)
# print(r.X)
r.initialize()
r.conditional_propability_of_y()
