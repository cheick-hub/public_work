## Periodic scripts

 1. `server/src/update_recent_papers.py`: Downloads most recent articles, computes their embeddings and saves everything.
 2. `server/src/update_all_papers.py`: Uses the data downloaded by the script above and updates the historical data, initially containing every paper of 2021.
 3. `server/src/update_model.py`: Uses the historical data (updated by the script just above) to update our clustering model.
 4. `server/src/update_front_end_data.py`: Updates the data to be displayed on the website. This preprocessing allows for a small response time of the server.

 Those four scripts are run in this order, forever, thanks to the script `server/update_all_data.sh`, combined with a systemd service.

## Run the server

You must have flask installed with Python.

Then, go in the `server` directory and run the server with the command `python -m flask run` (add `--host 0.0.0.0` if you want to open the server to the world, through your port 5000).