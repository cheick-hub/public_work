import arxivscraper
import datetime
import numpy as np
import pandas as pd
from util import embed_texts

def save_last_day_articles():
    categories = ['cs', 'econ', 'eess', 'math', 'physics', 'q-bio', 'q-fin', 'stat']
    papers = []
    seen_ids = set()
    for category in categories:
        print(f'Scraping category: {category}...')
        today = datetime.date.today()
        scraper = arxivscraper.Scraper(category=category, date_from=str(today - datetime.timedelta(days=2)), date_until=str(today))
        temp = scraper.scrape()
        print(temp)
        for paper in temp:
            if paper['id'] in seen_ids:
                continue
            papers.append(paper)
            seen_ids.add(paper['id'])
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(papers, columns=cols)
    title_embeds = embed_texts(df.title)
    abstract_embeds = embed_texts(df.abstract)
    df.to_json('data/recent_papers.json', orient='records', indent=2)
    np.save('data/recent_papers_abstract_embeddings.npy', abstract_embeds)
    np.save('data/recent_papers_title_embeddings.npy', title_embeds)

if __name__ == '__main__':
    save_last_day_articles()
