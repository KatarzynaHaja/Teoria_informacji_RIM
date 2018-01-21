import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = datasets.load_iris()
from sklearn.datasets.samples_generator import make_blobs
from scipy.optimize import minimize
X = iris.data
y = iris.target
# X, y = make_blobs(n_samples=100, centers=3, n_features=2,random_state=0)


class RIM:
    def __init__(self,X, y, K,alfa):
        self.K = K
        self.X = X
        self.y = y

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
        self.kmeans = KMeans(n_clusters=self.K).fit(self.X)

    def sigmoid(self, y, x,lambdas):
        # lambdas = lambdas.reshape(len(self.lambdas),self.X.shape[1])
        return np.exp(np.dot(y,x))/np.sum([np.exp(np.dot(i,x)) for i in lambdas])

    def delta(self,k,y):
        if k ==y:
            return 1
        else:
            return 0


    def derivative(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas), self.X.shape[1])
        derivatives = []
        for lam in range(len(lambdas)):
            der = []
            for d in range(self.X.shape[1]):
                s=0
                for i in range(len(self.X)):
                    y = self.kmeans.labels_[i]
                    s+=1.0/self.sigmoid(lambdas[y],self.X[i],lambdas) * self.sigmoid(lambdas[lam],self.X[i],lambdas) * self.X[i][d] * (self.delta(lam,y)- self.sigmoid(lambdas[y],self.X[i],lambdas))

                if d!=0:
                    s=-s+ self.alfa * 2 * lambdas[lam][d]
                else:
                    s=-s
                der.append(s)
            derivatives.append(np.array(der))
        # print(np.array(derivatives).ravel())
        return np.array(derivatives).ravel()

    def cost_function(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas),self.X.shape[1])
        reg = 0
        for i in range(len(lambdas)):
            for j in range(1, self.X.shape[1]):
                reg += lambdas[i][j] ** 2

        reg = self.alfa*reg
        result = 0
        for i in range(len(self.X)):
          y = self.kmeans.labels_[i]
          result+=np.log(self.sigmoid(lambdas[y],self.X[i],lambdas))
        return -(result-reg)

    def find_max(self):
        self.lambdas_ent =  minimize(self.cost_function,self.lambdas , method="BFGS",jac=self.derivative).x
        print( minimize(self.cost_function,self.lambdas , method="BFGS",jac=self.derivative))
        self.lambdas_ent = self.lambdas_ent.reshape(self.K, self.X.shape[1])


    def conditional_entropy(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas), self.X.shape[1])
        entropy =0
        for i in range(len(self.X)):
            for j in range(len(lambdas)):
                entropy +=self.sigmoid(lambdas[j],self.X[i],lambdas)*(-np.log(self.sigmoid(lambdas[j],self.X[i],lambdas)))
        return entropy/len(self.X)

    def entropy(self,lambdas):
        entropy =0
        for j in range(len(lambdas)):
            entropy+=self.prob_class(j,lambdas) * (-np.log(self.prob_class(j,lambdas)))
        return entropy

    def prob_class(self,y,lambdas):
        prob =0
        for i in range(len(self.X)):
            prob += self.sigmoid(lambdas[y],self.X[i],lambdas)
        return prob/len(self.X)

    def cost_function_rim(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas), self.X.shape[1])
        reg=0
        for i in range(len(lambdas)):
            for j in range(1,self.X.shape[1]):
                reg+=lambdas[i][j]**2
        reg = self.alfa * reg
        return -(self.entropy(lambdas) - self.conditional_entropy(lambdas) - reg)

    def derivative_rim(self,lambdas):
        lambdas = lambdas.reshape(len(self.lambdas), self.X.shape[1])
        derivative = []
        pc = [self.prob_class(c,lambdas) for c in range(len(lambdas))]
        for k in range(len(lambdas)):
            pk = self.prob_class(k, lambdas)
            der = []
            for d in range(self.X.shape[1]):
                s = 0

                for i in range(len(self.X)):
                    sigm = self.sigmoid(lambdas[k], self.X[i], lambdas)
                    s+=self.X[i][d] * sigm * (np.log(sigm / pk) - np.sum([self.sigmoid(lambdas[c], self.X[i],
                                                                                         lambdas) * np.log(
                    self.sigmoid(lambdas[c], self.X[i], lambdas) / pc[c]) for c in range(len(lambdas))]))
                if d!=0:
                    s=-s+ 2* self.alfa * self.lambdas[k][d]
                else:
                    s=-s
                der.append(s/len(self.X))
            derivative.append(np.array(der))
        return np.array(derivative).ravel()

    def find_max_Rim(self):
        self.lambdas_rim = minimize(self.cost_function_rim,self.lambdas_ent , method="BFGS",jac=self.derivative_rim).x
        print(minimize(self.cost_function_rim,self.lambdas_ent , method="BFGS",jac=self.derivative_rim))
        self.lambdas_rim = self.lambdas_rim.reshape(self.K, self.X.shape[1])


    def run(self):
        self.initialize()
        print(normalized_mutual_info_score(self.y, self.kmeans.labels_))
        print("poczatkowe",self.lambdas)
        self.find_max()
        print("po maxEnt",self.lambdas_ent)
        self.find_max_Rim()
        print("po rim",self.lambdas_rim)
        classes = self.choose_class()
        print(normalized_mutual_info_score(self.y, classes))

    def choose_class(self):
        tab = []
        self.classes =[]
        for i in self.X:
            for j in range(self.K):
                tab.append(self.sigmoid(self.lambdas[j],i,self.lambdas_rim))
        tab = np.array(tab).reshape(self.X.shape[0],self.K)
        for i in tab:
            print(np.argmax(i))
            self.classes.append(np.argmax(i))
        return self.classes
def draw(X,y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    plt.show()

r = RIM(X,y,2,0.2)
r.run()
draw(X,y)
draw(X,r.classes)
