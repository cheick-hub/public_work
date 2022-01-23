from flask import Flask, render_template, url_for   
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

def load_abstract():
    with open('data/to_display_abstract.json') as f:
        to_display_json = json.load(f)
    return to_display_json


def load_title():
    with open('data/to_display_title.json') as f:
        to_display_json = json.load(f)
    return to_display_json


# ROUTES
# ======

@app.route('/')
def default():
    return abstract()

@app.route("/abstract")
def abstract():
    return render_template("home.html", data=load_abstract(), title='title', link='title')


@app.route("/title")
def title():
    return render_template("home.html", data=load_abstract(), title='abstract', link='abstract')

