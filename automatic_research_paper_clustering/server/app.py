from flask import Flask
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

def load_articles():
    with open('data/to_display.json') as f:
        to_display_json = json.load(f)
    s = ''
    for cluster in to_display_json:
        s += f'<h2>{cluster}</h2>\n<p>'
        for article in to_display_json[cluster]:
            s += f'<a href="https://arxiv.org/abs/{article["id"]}">{article["title"]}</a><br/>\n'
        s += '</p>'
    return s


@app.route("/")
def hello_world():
    s = '<h1>Most frequent article clusters of the last 48 hours</h1>\n\n'

    s += load_articles()
    
    return s