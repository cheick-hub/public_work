from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def get_clusteriser(texts, texts_embeddings, n_clusters=100):
    df = pd.DataFrame({'text': texts})
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)

    df['cluster'] = list(kmeans.predict(texts_embeddings))

    corpus = []
    for cluster in range(kmeans.n_clusters):
        corpus.append(' '.join(df.text[df.cluster == cluster]))
    
    vectoriser = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    X = np.array(vectoriser.fit_transform(corpus).todense())

    cluster_names = [' ; '.join(e) for e in np.array(vectoriser.get_feature_names())[X.argsort(axis=1)[:, ::-1][:, :5]]]

    clusters_df = pd.DataFrame({'cluster_id': list(range(clusteriser.n_clusters)), 'cluster_name': cluster_names}).set_index('cluster_id')

    return kmeans, clusters_df

def embed_texts(texts: list):
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return model.encode(texts)
