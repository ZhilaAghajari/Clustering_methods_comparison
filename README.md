# Clustering_methods_comparison
---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    name: python3
---

```python colab={"base_uri": "https://localhost:8080/"} id="ciBzrC4sOu7-" outputId="7e6d7fc7-661c-4ba4-c3fc-c8337be93f71"
!gdown --id 1CACpD4cXHJCHDUlyS0Yih-ScP568eJvi
```

```python colab={"base_uri": "https://localhost:8080/"} id="6ehM-9JgO6is" outputId="9b32c8ae-97fa-4508-aefc-ff2cb8853ed4"
!gdown --id 1hq9Q9B6fvUPyazzrdnnqSy5JO7sAuUg4
```

```python id="bAzmUMmgO-Ol"
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import linalg
import math
from sklearn.cluster import KMeans, SpectralClustering # I have it here to verify my results .. 
from sklearn import metrics

```

```python id="4FylU1szP9l9" colab={"base_uri": "https://localhost:8080/"} outputId="9ebaf9c1-1e5e-49a4-c007-43020331ae98"
#read the dataset as numpy
#data = np.loadtxt(open('cho.txt',newline=''), delimiter='\t')
data = np.loadtxt(open('iyer.txt',newline=''), delimiter='\t')
data.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 288} id="AnGfEZuJa9Ca" outputId="75f77d64-59f1-4e47-986f-22fc5e78c5a3"
import pandas as pd

pd.DataFrame(data).describe()
```

```python colab={"base_uri": "https://localhost:8080/"} id="YsflvElQgyPu" outputId="0f5fa008-3657-4104-94ec-fbda74ff50f5"
#see the numbers of outlier in each data set:
(np.argwhere(data[:,1] == -1)).shape

```

```python colab={"base_uri": "https://localhost:8080/", "height": 298} id="MWpRojhRTeRK" outputId="10c4a494-fc38-4e83-cf40-b424a99d2433"
#visualize the data
#normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[:,2:])

#PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
plt.scatter(x = pca_data[:, 0], y = pca_data[:,1], c=data[:,1])
plt.title('PCA of Iyer dataset')

```

```python id="zN64kuNwZXP9"
#Kmeans clustering 
def Kmeans(data, K, eps = 0.001):
  #
  centers = np.zeros((K, len(data[0])))
  previous_centers = centers
  #choose random centers
  rand_indx = np.random.permutation(len(data))
  centers = data[rand_indx[0:K], :]

  #set the cluster labels as zeros first
  clusters = np.zeros((len(data)))
  center_changes=1
  while(center_changes >= eps):
    center_changes = 0
    for i in range(len(data)):
      dis = distance(data[i, :], centers)
      clusters[i] = np.argmin(dis)
    # I had to do deep copy here .. 
    previous_centers[:, :] = centers[:, :]

    #updating the clusters:
    for i in range(0, K):
      temp = data[np.argwhere(clusters==i),:]
      centers[i, :] = np.mean(temp, axis=0)
    #check if the centers changed noticably
    for i in range(0, K):
      center_changes += np.sqrt(sum(np.power(previous_centers[i, :]- centers[i, :], 2)))
    
  return(clusters,centers)


```

```python id="L_5Xva0DkN_J"

```

```python id="vSn_6cVji_9b"
def distance(data, centers):
  dis = [0]*len(centers)
  for i in range(len(centers)):
    dis[i] = np.sqrt(sum(np.power((data-centers[i]), 2)))
  return dis

```

<!-- #region id="1baxlhvnjsop" -->

<!-- #endregion -->

```python id="aTzsVYcGlxBP"
# K=4
# clusters, centers = Kmeans(data[:, 2:], K, eps = 0.0001)
#visualize the data
#PCA
def visualize(data, cluster, title):
  pca = PCA(n_components=2)
  pca_data = pca.fit_transform(data)
  plt.scatter(x =pca_data[:, 0], y = pca_data[:,1], c=clusters)
  plt.title(title)
```

```python id="4z4mwrUDln3-"
# external index --  I used entropy, purity and normalized mutual information as the external index 
def external_index(clusters, labels):
  entropy = 0
  total_purity = 0
  unique_labels = list(set(labels))
  # I analyzed the data once without outliers once with the outliers to see what happens
  if -1 in unique_labels:
    unique_labels.remove(-1) # we need to remove the outliers
  
  unique_clusters = list(set(clusters))

  for clstr in unique_clusters:
    entropy_i = 0
    purity_i =0
    purity_max = 0
    for l in unique_labels:
      Ci = [x for x in range(len(labels)) if labels[x]==l] # all the samples with class c
      Di = [x for x in range(len(clusters)) if clusters[x] == clstr] # samples within clstr cluster 

      P_i_j = len([x for x in Ci if x in Di])/len(Di) # 
      if P_i_j!=0:
        entropy_i += P_i_j*math.log2(P_i_j)
        # entropy_i = -1*entropy_i
        purity_i = max(P_i_j,purity_i)
    entropy += (len(Di)/len(clusters))*(entropy_i)
    total_purity += ((len(Di)/len(clusters))*(purity_i))
  entropy = (-1*entropy) #I need to ask whether I need to report it as a negative or positive value -- there is negative in the formula but it is mentioned it is a negative measure! Got it: it is a negative measure in a sense that the lower the value the better our results
  return entropy, total_purity




```

```python id="YnpxmggIFd8G"

```

```python id="uJbSf-_94R-D"
#internal index -- Sihousetee Coefficient
def internal_index(data, clusters, labels):
  sht = [0]*len(data)
  unique_clusters = list(set(clusters))
  a_i = 0
  for i in range(len(data)):
    clstr = int(clusters[i])
    if (len(clusters[clusters==clstr])-1)!=0:
      a_i = sum(distance(data[i, :], data[clusters==clstr, :]))/(len(clusters[clusters==clstr])-1)
    
    #b = [0]*len(unique_clusters)
    b = []
    for j in range(len(unique_clusters)):
      if int(unique_clusters[j] == clstr):
        continue
      b.append(sum(distance(data[i, :],data[clusters==unique_clusters[j],: ]))/len(clusters[clusters==unique_clusters[j]]))
    if len(b)>0:
      # print('what is B: ', b)
      b = min(b)
    if max(a_i,b)!=0:
      sht[i] = (b-a_i)/max(a_i,b)
    if np.isnan(sht[i]):
      print(max(a_i,b))
  SHT = np.mean(sht)
  # if np.isnan(SHT):
  #   print('check this: ',(b-a_i), max(a_i,b), sht)
  return SHT
```

```python id="RA6xZVoKpbUd"
#spectral clustering
def spectral(data, K, sigma):
  #W : the simialrity matrix
  W = np.zeros((len(data), len(data)))
  #D: Degree matrix
  D = np.zeros((len(data), len(data)))

  for i in range(len(data)):
    temp= np.exp((-1*(np.asarray(distance(data[i,:], data))))/2*sigma**2)
    # only choose the K-nearset neughbors (based on slide 58), and the rest will be zeros
    temp[temp<(sorted(temp)[len(temp)-K])] = 0 
    W[i, :] = temp
    D[i, i] = sum(temp) #sum of each row in the W matrix 
  L = D - W
  
  #get the eigne values and the eigen vectors
  eig_vector, eig_value = linalg.eig(L)
  # print('these are the eig_vector: ', eig_vector) -> So, eigen vectors are complex numbers!!
  #choose the K as the one that maximizes the difference between two sequential eignen vectors .. 
  #choosen_k = np.argmax(((np.sort_complex(eig_vector)[1:]) - (np.sort_complex(eig_vector)[0:-1]))) 
  #print('Chosen K: {0} , len of data {1}'.format(choosen_k,len(data)))
  choosen_k = len(data) -  np.argmax(np.sort_complex(eig_vector)[1:] - np.sort_complex(eig_vector)[0:-1]) #sort complex sort them in assending order but we need the descending order 
  # print('Chosen K: ', np.argmax(np.sort_complex(eig_vector)[1:] - np.sort_complex(eig_vector)[0:-1]) )
  print(len(data))

  indx = np.argsort(eig_vector)[-K:]
  egin_data = eig_value[:, indx]

  clusters, centers = Kmeans(egin_data, choosen_k)
  return clusters, centers
```

```python id="HUo57A6G6bgX"
def best_matching(k_labels, truth):
  k_labels_matched = np.empty_like(k_labels)
  # For each cluster label...
  for k in np.unique(k_labels):

      # ...find and assign the best-matching truth label
      match_nums = [np.sum((k_labels==k)*(truth==t)) for t in np.unique(truth)]
      k_labels_matched[k_labels==k] = np.unique(truth)[np.argmax(match_nums)]
  return k_labels_matched
```

```python id="R1cLmgVWN-lu"
#FOR VALIDATION:
# I only added this to validate my purity calculation. this is done using contingency matrix and is similar to formula given in slide 51 of lecture 8 
from sklearn.metrics import confusion_matrix
def purity_score(truth, y_pred):
    cm = confusion_matrix(truth, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 385} id="B_PqTwRnJpO4" outputId="59ca1a93-6384-4c68-d08f-33ae96c4a794"
# ************* << K-means resuls >> *************
K = 4
clusters, centers = Kmeans(data[:, 2:], K)
k_labels_matched = best_matching(clusters, data[:, 1])
visualize(data[:, 2:], k_labels_matched, 'Kmeans with 4 class')
entropy, purity = external_index(k_labels_matched, data[:, 1])
SHT = internal_index(data, k_labels_matched, data[:, 1])
print(' ---- K-means results ----')
print('External indexs:')
print('Entropy: ',entropy)
print('purity: ',purity)
print('Sihousetee (internal_index): ', SHT)

print('Validating purity using confusion matrix: ', purity_score(data[:, 1], k_labels_matched) )


```

```python id="DWD5T_TFylhN"

```

```python id="tREqikZw6G_8"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 436} id="qqGUfZAIL7-f" outputId="8f79d4f3-a4ad-4c95-d180-b87302b11edf"
# Spectral method
K = 4
sigma = 1
clusters, centers = spectral(data[:, 2:], K, sigma)
k_labels_matched = best_matching(clusters, data[:, 1])
k_labels_matched = clusters
visualize(data[:, 2:], k_labels_matched, 'Spectral clustering')
entropy, purity = external_index(k_labels_matched, data[:, 1])
print(k_labels_matched.shape)
SHT = internal_index(data[:, 2:], k_labels_matched, data[:, 1])
print(' ---- Spectral results ----')
print('External indexs:')
print('Entropy: ',entropy)
print('purity: ',purity)
print('Sihousetee (internal_index): ', SHT)
print('Validating purity using confusion matrix: ', purity_score(data[:, 1], k_labels_matched) )





```

```python colab={"base_uri": "https://localhost:8080/", "height": 367} id="AvCNLLDWiBz1" outputId="2be9f102-a719-4dd8-b777-dd204dbe5a51"
#VALIDATION:
# Here I am only checking my result with Sklearn results:
clusters_sklearn = KMeans(n_clusters = 4).fit(data[:, 2:])

visualize(data[:, 2:], clusters_sklearn, 'K means SK-learn')
entropy, purity = external_index(clusters_sklearn.labels_, data[:, 1])
# print('shape of clustersss ', clusters_sklearn.labels_.shape)
# print('shape of labels ', data[:, 2].shape)
SHT = internal_index(data, clusters_sklearn.labels_, data[:, 1])
print(' ---- K-means results ----')
print('External indexs:')
print('Entropy: ',entropy)
print('purity: ',purity)
print('Sihousetee (internal_index): ', SHT)




```

```python colab={"base_uri": "https://localhost:8080/", "height": 436} id="OH1krwIhlF5E" outputId="b56b2709-1ea0-420a-f8c3-a5e3d411e8d5"
# VALIDATION:
#checking my results with SpectralClustering from the Sklearn:
clusters_sklearn = SpectralClustering(n_clusters = 4).fit(data[:, 2:])
visualize(data[:, 2:], clusters_sklearn, 'SpectralClustering from the Sklearn')
entropy, purity = external_index(clusters_sklearn.labels_, data[:, 1])
SHT = internal_index(data, clusters_sklearn.labels_, data[:, 1])
print(' ---- SpectralClustering results ----')
print('External indexs:')
print('Entropy: ',entropy)
print('purity: ',purity)
print('SHT ', SHT)
```

```python id="LayUQgTHOMV1"
#analysing the two methods on different values of K
def exploring_k_effects(data):
  itr = 30
  results = np.zeros((itr, 4))
  for k in range(2,itr):
    # clusters, centers = Kmeans(data, K)
    # clusters, centers = spectral(data, K, 1)
    clusters, centers = Kmeans(data[:, 2:], k)
    k_labels_matched = best_matching(clusters, data[:, 1])

    entropy, purity = external_index(k_labels_matched, data[:, 1])
    SHT = internal_index(data, k_labels_matched, data[:, 1])
    results[k, 0] = k
    results[k, 1] = entropy
    results[k, 2]= purity
    results[k, 3] = SHT


  return results 

results = exploring_k_effects(data)



```

```python colab={"base_uri": "https://localhost:8080/", "height": 298} id="5aeQuITexI5O" outputId="dea2ac7d-68ba-44c1-afb2-f03e616afc02"
#ploting
plt.plot(results[2:,0],results[2:,1])
plt.title('the effect of K value on entropy')

# plt.plot(results[2:,0],results[2:,2])
# plt.title('K effects on Purity')
# plt.show()

# plt.plot(results[2:,0],results[2:,3])
# plt.title('the effect of K value on Sihousetee Coefficient')
```
