from flask import Flask,jsonify,render_template
#from os.path import join
from os import listdir

MODELS_DIR = "./static/models/"

app = Flask(__name__)
@app.route('/')
def main():
    return render_template('index.html')
@app.route('/api/v1/models')
def getModelsFiles():
    return jsonify(listdir(MODELS_DIR))
    
    