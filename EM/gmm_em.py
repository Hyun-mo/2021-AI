import numpy as np
from scipy.stats import multivariate_normal as mvn #확률밀도 함수
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

class GMM:
    def __init__(self, K, n_runs, X):
        self.X = X # 전체 데이터
        self.K = K # 군집 수
        self.n_runs = n_runs # 최대 반복 횟수
        self.labels = np.random.randint(0,3,(self.X.shape[0])) # 초기 라벨 벡터
        self.score = 0 # 실루엣 스코어
        self.count = 0 # 실루엣 지표가 변하지 않으면 증가
        #self.mu, self.sigma, self.pi =  self.initialise() # 랜덤 초기화 시작
        self.mu, self.sigma, self.pi =  self.initialise_kmeans() # kmeans의 결과로 초기화 후 시작
        #초기화 이후 실루엣 스코어와 군집상태 그리기
        self.get_silhouette_score()
        self.draw('initial, score: %f' %self.score)
    
        
    def initialise(self):
        d = self.X.shape[1] # 2차원 데이터
        labels = np.unique(self.labels) # 라벨 벡터
        means = np.zeros((self.K, d)) # 평균 행렬
        sigma = np.zeros((self.K, d, d)) # 공분산 행렬
        pi = np.zeros(self.K) # 파이

        counter=0
        for label in labels:
            ids = np.where(self.labels == label)
            pi[counter] = len(ids[0]) / self.X.shape[0]
            means[counter,:] = np.mean(self.X[ids], axis = 0)
            de_meaned = self.X[ids] - means[counter,:]
            Nk = self.X[ids].shape[0]
            sigma[counter,:, :] = np.dot(pi[counter] * de_meaned.T, de_meaned) / Nk
            counter+=1
        assert np.sum(pi) == 1
        
        return means, sigma, pi
    
    #kmeans로 초기 라벨링
    def initialise_kmeans(self):
        n_clusters = self.K
        kmeans = KMeans(n_clusters= n_clusters, max_iter=3)
        kmeans.fit(self.X)
        self.labels = kmeans.predict(self.X)
        means, sigma, pi = self.initialise()

        return means, sigma, pi
            
    def e_step(self):
        N = self.X.shape[0] #데이터 수
        self.gamma = np.zeros((N, self.K))
        
        for c in range(self.K):
            self.gamma[:,c] = self.pi[c] * mvn.pdf(self.X, self.mu[c,:], self.sigma[c])

        gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma
    
    
    def m_step(self):
        C = self.gamma.shape[1] # 군집 수

        self.pi = np.mean(self.gamma, axis = 0)
        self.mu = np.dot(self.gamma.T, self.X) / np.sum(self.gamma, axis = 0)[:,np.newaxis]

        for c in range(C):
            x = self.X - self.mu[c, :] 
            gamma_diag = np.diag(self.gamma[:,c])
            gamma_diag = np.matrix(gamma_diag)
            sigma_c = x.T * gamma_diag * x
            self.sigma[c,:,:]=(sigma_c) / np.sum(self.gamma, axis = 0)[:,np.newaxis][c]

        return self.pi, self.mu, self.sigma
    
    #sklearn의 실루엣 스코어
    def get_silhouette_score(self):
        self.predict()
        last = self.score
        self.score = silhouette_score(self.X, self.labels)
        if last == self.score:
            self.count += 1
        else:
            self.count = 0
        return self.score
    
    #em 알고리즘 반복
    def fit(self):
        try:
            for run in range(self.n_runs):  
                self.e_step()
                self.m_step()
                self.get_silhouette_score()
                self.draw(title ='run: %02d, score: %f' %(run, self.score))
                if self.count > 5:
                    break
        except Exception as e:
            print(e)

        return self
    
    #파라미터들을 기준으로 군집 예측
    def predict(self):
        labels = np.zeros((self.X.shape[0], self.K))
        for c in range(self.K):
            labels [:,c] = self.pi[c] * mvn.pdf(self.X, self.mu[c,:], self.sigma[c])
        
        self.labels  = labels.argmax(1)
        return self.labels 

    #예측한 군집들을 그래프에 그리기
    def draw(self, title):
        ax.clear()
        ax.set_title('%s' %title, fontweight="bold")
        ax.scatter(self.X[:, 0], self.X[:, 1],c=self.labels)
        plt.draw()
        plt.savefig('./result/%s.png' %title, dpi=400)
        plt.pause(0.1)

if __name__ == '__main__':
    X = np.loadtxt('./points.csv', delimiter=',')
    global fig, ax 
    fig, ax = plt.subplots()
    print('start')
    start = time.time()
    model = GMM(K=3, n_runs = 100, X=X)
    fitted_values = model.fit()
    end = time.time()-start
    print(end)
    predicted_values = model.predict()
    ax.set_title('score:%f\ntime:%f' %(model.get_silhouette_score(), end))
    ax.scatter(X[:, 0], X[:, 1],c=predicted_values)
    plt.show()