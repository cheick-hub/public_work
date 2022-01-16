import pandas as pd
import numpy as np
import json
import pickle

def update():
    recent_df = pd.read_json('data/recent_papers.json', orient='records')
    recent_embeddings = np.load('data/recent_papers_embeddings.npy')
    with open('model/clusteriser.model', 'rb') as f:
        kmeans = pickle.load(f)
    clusters_names = np.array(list(pd.read_csv('model/cluster_names.csv').cluster_name))
    clusters = kmeans.predict(recent_embeddings)
    clusters_count = {}
    for cluster in clusters:
        clusters_count[cluster] = clusters_count.get(cluster, 0) + 1
    clusters_count = list(clusters_count.items())
    clusters_count.sort(key=lambda e: e[1], reverse=True)
    most_frequent_clusters = [e[0] for e in clusters_count]
    res = {}
    for i, cluster in enumerate(most_frequent_clusters):
        cluster_name = f'#{i+1}: {clusters_names[cluster]}:'
        res[cluster_name] = []
        cluster_df = recent_df.loc[clusters == cluster]
        for j, row in cluster_df.iterrows():
            id = row.id
            title = row.title
            res[cluster_name].append({'id': id, 'title': title.capitalize().replace('  ', '')})
    with open('data/to_display.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    update()