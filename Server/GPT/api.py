from flask import Flask,jsonify,request#,abort
from DevChat import *
from os import listdir
app = Flask(__name__)

class ChatInstance(Chat):
    # implementation of singleton pattern
    instance = None
    def __new__(cls,**kwargs):
        if(not(hasattr(cls,'instance'))):
            cls.instance = super(ChatInstance,cls).__new__(cls)
        return cls.instance
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    @classmethod
    def reloadModel(cls,args):
        del(cls.instance)        
        cls.instance = ChatInstance(**args)
    @staticmethod
    def loadBotSettings():
        with ModelLoader("bot_settings.json",ChatBotSettings) as model:
            chatSettings = model
        return chatSettings
    @staticmethod
    def saveBotSettings(settings:ChatBotSettings):
        with open("./config/bot_settings.json",'w') as wr:
            wr.write(str(settings))
chat_args = {"message_history":[]}
chat_instance = Chat(**chat_args)

@app.route("/api/models")# get models
def getModels():
    return jsonify(listdir('./model'))# get all the models
@app.route("/api/<model>")# set the model
def setModel(model):
    if(not(model in listdir('./model'))):
        #abort(404);
        return jsonify({"Error":"Unikown Model!"})
    settings = ChatInstance.loadBotSettings()
    settings.model_path = model
    ChatInstance.saveBotSettings(settings)
    ChatInstance.reloadModel(chat_args)
    return {"status":200}
@app.route("/api/<text>")# input text
def generateText(text:str):
    return ''
@app.route("/api/intimacy")
def getIntimacy(text:str):
    # para el nivel de intimidad
    return ''
@app.route('/api/live2dModel')
def getLive2dModel():
    # indicara que modelo de live2d cargar
    # dependiendo del estado de animo de la waifu
    return 'goth'



