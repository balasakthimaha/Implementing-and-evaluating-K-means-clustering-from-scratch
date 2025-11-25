import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    centroids=[]
    for i in range(k):
        pts=X[labels==i]
        centroids.append(pts.mean(axis=0) if len(pts)>0 else X[np.random.choice(len(X))])
    return np.array(centroids)

def compute_inertia(X, centroids, labels):
    return sum(np.linalg.norm(X[i]-centroids[labels[i]])**2 for i in range(len(X)))

def kmeans(X,k,max_iters=100,tol=1e-4):
    centroids=initialize_centroids(X,k)
    for _ in range(max_iters):
        labels=assign_clusters(X,centroids)
        new=update_centroids(X,labels,k)
        if np.linalg.norm(new-centroids)<tol: break
        centroids=new
    return centroids,labels,compute_inertia(X,centroids,labels)

if __name__=='__main__':
    np.random.seed(0)
    c1=np.random.randn(100,2)
    c2=np.random.randn(100,2)+np.array([5,5])
    c3=np.random.randn(100,2)+np.array([10,0])
    X=np.vstack([c1,c2,c3])
    k=3
    centroids,labels,inertia=kmeans(X,k)
    print('Final Inertia:',inertia)
    plt.scatter(X[:,0],X[:,1],c=labels)
    plt.scatter(centroids[:,0],centroids[:,1],marker='X',s=200)
    plt.savefig('clusters.png')
