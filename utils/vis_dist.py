
import numpy as np
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

current_path = "/mnt/Data3/hanqiyan/style_transfer/"

domain_dict = ["imdb","yelp_dast","amazon"]

def visualize_dist(senti=1):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    # pca = PCA(n_components=50)
    all_vectors,all_labels = [],[]
    for domid,domain_name in enumerate(domain_dict):
        senti_labels = np.load(current_path+"%s.npy"%domain_name)
        # senti_idx = np.where(senti_labels==senti)
        senti_idx = np.arange(senti_labels.shape[0])
        domain_vectors = np.load(current_path+"%s_contextualVec.npy"%domain_name)[senti_idx]
        domain_labels = senti_idx.shape[0]*[domid]
        all_vectors.append(domain_vectors)
        all_labels.extend(domain_labels)
        # pca_result = pca.fit_transform(domain_vectors)
    z = tsne.fit_transform(np.concatenate(all_vectors,axis=0))
    df = pd.DataFrame()
    df["y"] = all_labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
            palette=sns.color_palette("hls",len(domain_dict)),
            data=df).set(title="MNIST data T-SNE projection")
    plt.savefig("tsne_%s.png"%domain_name)
    return None

visualize_dist(senti=0)