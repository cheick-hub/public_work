import numpy as np
import pandas as pd

def update_all_papers():
    historical_df = pd.read_json('data/all_papers.json', orient='records')
    historical_abstract_embeddings = np.load('data/all_papers_abstract_embeddings.npy')
    historical_title_embeddings = np.load('data/all_papers_title_embeddings.npy')

    recent_df = pd.read_json('data/recent_papers.json', orient='records')
    recent_abstract_embeddings = np.load('data/recent_papers_abstract_embeddings.npy')
    recent_title_embeddings = np.load('data/recent_papers_title_embeddings.npy')

    historical_ids = set(historical_df.id)
    is_new = list(recent_df.id.apply(lambda id: id not in historical_ids))
    new_df = recent_df.iloc[is_new]
    new_abstract_embeddings = recent_abstract_embeddings[is_new]
    new_title_embeddings = recent_title_embeddings[is_new]

    updated_df = pd.concat((historical_df, new_df))
    updated_abstract_embeddings = np.vstack((historical_abstract_embeddings, new_abstract_embeddings))
    updated_title_embeddings = np.vstack((historical_title_embeddings, new_title_embeddings))

    updated_df.to_json('data/all_papers.json', orient='records', indent=2)
    np.save('data/all_papers_abstract_embeddings.npy', updated_abstract_embeddings)
    np.save('data/all_papers_title_embeddings.npy', updated_title_embeddings)

if __name__ == '__main__':
    update_all_papers()
