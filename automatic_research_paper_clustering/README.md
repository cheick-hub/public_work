This repository contain all the necessary in order to automatically cluster research paper.

## Periodic scripts

`/server/src/recent_papers_scraper.py`: Downloads most recent articles, computes their embeddings and saves everything.

`/server/src/update_data.py`: Uses the data downloaded by the script above and updates the historical data, initially containing every paper of 2021.

`/server/src/update_model.py`: Uses the historical data (updated by the script just above) to update our the clustering model.

## Run the server

WIP