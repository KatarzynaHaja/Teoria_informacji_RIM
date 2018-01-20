import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = datasets.load_iris()
from scipy.optimize import minimize
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
        self.lambdas = np.random.rand(K, self.X.shape[1])

    def normalize(self):
        means = np.mean(self.X, axis = 0)
        stds = np.std(self.X, axis = 0)
        self.X = (self.X - means)/stds

    def initialize(self):
        self.kmeans = KMeans(n_clusters=self.K,init="k-means++").fit(self.X)
        for i in range(len(self.X)):
            self.cluster_by_k_means[self.kmeans.labels_[i]].append(self.X[i])

    def test(self):
        acc = accuracy_score(self.y, self.kmeans.labels_)
        mi = normalized_mutual_info_score(self.y, self.kmeans.labels_)


    def sigmoid(self, y, x,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas),self.X.shape[1])
        return np.exp(np.dot(y,x))/np.sum([np.exp(np.dot(i,x)) for i in lambdas])

    def delta(self,k,y):
        if k ==y:
            return 1
        else:
            return 0

    def derivative(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas),self.X.shape[1])
        derivatives = []
        for lam in range(len(lambdas)):
            d = []
            for i in range(len(self.X)):
              y = self.kmeans.labels_[i]
              d.append(1.0/self.sigmoid(lambdas[y],self.X[i],lambdas) * self.sigmoid(lambdas[lam],self.X[i],lambdas) * self.X[i] * (self.delta(lam,y)- self.sigmoid(lambdas[y],self.X[i],lambdas)- self.alfa *2 * lambdas[lam] ))

            derivatives.append(-np.sum(np.array(d), axis =0))
        return np.array(derivatives).ravel()

    def cost_function(self,lambdas):
        p =[]
        lambdas = lambdas.reshape(len(self.lambdas),self.X.shape[1])
        for i in range(len(self.X)):
          y = self.kmeans.labels_[i]
          p.append(np.log(self.sigmoid(y,self.X[i],lambdas)))
        return -np.sum(p)

    def find_max(self):
        self.lambdas = minimize(self.cost_function,self.lambdas , method="Newton-CG", jac=self.derivative).x.reshape(self.K, self.X.shape[1])
        self.lambdas = self.lambdas.reshape(self.K, self.X.shape[1])

    def conditional_entropy(self,lambdas):
        entropy =0
        for j in range(len(lambdas)):
            for i in range(len(self.X)):
                entropy +=self.sigmoid(lambdas[j],self.X[i],lambdas)*(-np.log(self.sigmoid(lambdas[j],self.X[i],lambdas)))
            entropy = 1.0/len(self.X)*entropy
        return entropy

    def entropy(self,lambdas):
        entropy =0
        for j in range(len(lambdas)):
            entropy+=self.prob_class(lambdas[j],lambdas) * np.log(self.prob_class(lambdas[j],lambdas))
        return -entropy

    def prob_class(self,y,lambdas):
        prob =0
        for i in range(len(self.X)):
            prob += self.sigmoid(y,self.X[i],lambdas)
        return 1.0/len(self.X)*prob

    def cost_function_rim(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas), self.X.shape[1])
        return -(self.entropy(lambdas) - self.conditional_entropy(lambdas))

    def derivative_rim(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas), self.X.shape[1])
        derivative = []
        for j in range(len(lambdas)):
            s = []
            cl = self.prob_class(lambdas[j], lambdas)
            for i in range(len(self.X)):
                sigm = self.sigmoid(lambdas[j],self.X[i],lambdas)
                z = np.array(self.X[i,0]*sigm * (np.log(sigm/cl)- np.sum([self.sigmoid(lambdas[z],self.X[i],lambdas) * np.log(self.sigmoid(lambdas[z],self.X[i],lambdas)/cl) for z in range(len(lambdas))])))
                rest = self.X[i,1:]*sigm * (np.log(sigm/cl)- np.sum([self.sigmoid(lambdas[z],self.X[i],lambdas) * np.log(self.sigmoid(lambdas[z],self.X[i],lambdas)/cl) for z in range(len(lambdas))]) * 2* self.alfa * self.lambdas[j,1:])
                s.append(np.append(z,rest))

            derivative.append(-np.mean(s,axis=0))
        return np.array(derivative).ravel()

    def find_max_Rim(self):
        self.lambdas = minimize(self.cost_function_rim,self.lambdas , method="Newton-CG", jac=self.derivative_rim).x.reshape(self.K, self.X.shape[1])
        self.lambdas = self.lambdas.reshape(self.K, self.X.shape[1])

    def run(self):
        self.initialize()
        print(self.lambdas)
        self.find_max()
        print(self.lambdas)
        self.find_max_Rim()
        print("duuup",self.lambdas)
        print(self.choose_class())

    def choose_class(self):
        tab = []
        classes =[]
        for i in self.X:
            for j in range(self.K):
                tab.append(self.sigmoid(self.lambdas[j],i,self.lambdas))
        tab = np.array(tab).reshape(self.X.shape[0],self.K)
        for i in tab:
            classes.append(np.argmax(i))
        return classes
def draw():
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    plt.show()

r = RIM(X,y,3,10)
r.run()
# r.initialize()
# #print(r.derivative(r.lambdas))
# # draw()
# # print(r.find_max())
# print(r.derivative_rim(r.lambdas))
