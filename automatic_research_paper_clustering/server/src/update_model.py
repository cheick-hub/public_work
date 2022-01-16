import numpy as np
from sklearn.cluster import KMeans
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def update_model():

    # Creating KMeans
    N_CLUSTERS = 100
    embeddings = np.load('data/all_papers_embeddings.npy')
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    kmeans.fit(embeddings)
    with open('model/clusteriser.model', 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Creating cluster labels
    df = pd.read_json('data/all_papers.json', orient='records')
    df['cluster'] = list(kmeans.predict(embeddings))
    corpus = []
    for cluster in range(N_CLUSTERS):
        corpus.append(' '.join(df.abstract[df.cluster == cluster]))
    vectoriser = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    X = np.array(vectoriser.fit_transform(corpus).todense())
    cluster_names = [' ; '.join(e) for e in np.array(vectoriser.get_feature_names())[X.argsort(axis=1)[:, ::-1][:, :5]]]
    clusters_df = pd.DataFrame({'cluster_id': list(range(N_CLUSTERS)), 'cluster_name': cluster_names}).set_index('cluster_id')
    clusters_df.to_csv('model/cluster_names.csv')

if __name__ == '__main__':
    update_model()