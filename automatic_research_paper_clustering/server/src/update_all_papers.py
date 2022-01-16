import numpy as np
import pandas as pd

def update_all_papers():
    historical_df = pd.read_json('data/all_papers.json', orient='records')
    historical_embeddings = np.load('data/all_papers_embeddings.npy')

    recent_df = pd.read_json('data/recent_papers.json', orient='records')
    recent_embeddings = np.load('data/recent_papers_embeddings.npy')

    historical_ids = set(historical_df.id)
    is_new = list(recent_df.id.apply(lambda id: id not in historical_ids))
    new_df = recent_df.iloc[is_new]
    new_embeddings = recent_embeddings[is_new]

    updated_df = pd.concat((historical_df, new_df))
    updated_embeddings = np.vstack((historical_embeddings, new_embeddings))

    updated_df.to_json('data/all_papers.json', orient='records', indent=2)
    np.save('data/all_papers_embeddings.npy', updated_embeddings)

if __name__ == '__main__':
    update_all_papers()