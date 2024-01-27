
from dataclasses import dataclass
from typing import ClassVar
from json import dumps,loads,JSONDecodeError
from os.path import join,isfile,isdir
from os import listdir
from uuid import uuid4
from datetime import datetime
import chromadb
# from os import getcwd

convertObject2JsonData = lambda self: dumps(dict((x,y) for x,y in vars(self).items() if not(x.startswith("_"))),indent=4)
convert2Dict = lambda self: dict((x,y) for x,y in vars(self).items() if not(x.startswith("_")))
# use this method inside the __str__


# MERGER

class ModelLoader:
    '''
    this class will convert the JSON config file to a Model like an ORM
    in this case implements the DAO Pattern
    '''
    # settings_path:str = join(getcwd(),'Server','GPT','config/') testing
    settings_path:str = 'config/'# the path of the configurations
    def __init__(self,configuration_name:str,ModelClass,no_join_config_file_path=False):
        '''
            configuration_name -> name of the config file
        no_join_config_file_path -> donot use join it is usseful when you are using a model
        that save the data in other directori diferent to config
        '''
        self.settings_path = join(ModelLoader.settings_path,configuration_name)
        if no_join_config_file_path:
            self.settings_path = configuration_name
        self.ModelClass = ModelClass
    def exceptions_processor(self,**kwargs):
        if(kwargs.get("exec_val",False)):
            raise NameError(f"{' '.join([ x for x in kwargs])}\n")
    def convertEmbebedClass(self):
        """this method will check if the model has any embebed model"""
        if hasattr(self.ModelClass,'_metha_info'):
            # si se tiene unicamente
            if self.ModelClass._metha_info.get('embebed_models',False):
                # si existe en el diccionario de descripcion
                for subModel in self.ModelClass._metha_info.get('embebed_models'):
                    if not(hasattr(subModel,'_for')): raise NameError("Atributo _for no definido en clase {subModel.__name__} o")
                    req = self.configs.get(subModel._for,False)
                    #print('rq',subModel._for)
                    if not(req): raise NameError(f'Atributo no encontrado en el archivo de configuracion !')
                    for kconf in self.configs.keys():
                        if subModel._for == kconf:
                            self.configs[kconf] = subModel(**self.configs[kconf])# convertimos
                            
                            # google_trans == google_trans 
                
                
    def __enter__(self):
        if not(isfile(self.settings_path)):
            raise ValueError(f"File not found ! \n create the file ! error path: {self.settings_path}")
        self.file_raw = open(self.settings_path,'r')
        try:
            self.configs = loads(self.file_raw.read())
        except JSONDecodeError as e:
            raise NameError(f'BAD CONFIG FILE ! \n {e}')
        try:
            self.convertEmbebedClass()
            self.configs = self.ModelClass(**self.configs)
            # try to convert into ModelClass
        except Exception as e:
            raise NameError(f"error in {e} while converting configs to Class Model")
        return self.configs
    def __exit__( self, exc_type, exc_val, exc_tb ):
        # exceptions 
        self.exceptions_processor(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)
        self.file_raw.close()
    
# DATA CLASSES

@dataclass
class ChromaDbClient:
    _for = "chroma_config"
    host:str = "localhost"
    port:int = 8000
    only_http:bool = False# pip install chromadb-client https://docs.trychroma.com/usage-guide
    def __str__(self):
        return convertObject2JsonData(self)

    
    
@dataclass
class ChromaDb:
    # class attrs
    availableChromaDbModes:ClassVar[list[str]] = ["client","http_client","server_process"]
    ''' 
    
    choma modes descriptions:
        client -> in this mode the chomadb will be executed in the current process it allows local storage 
        server_process -> this mode executes a chomadb server in an other process and the current process will connect to them
        http_client -> this chomadb mode executes a http server and the client will conect to them
        
    '''
    #metha attrs
    _metha_info:ClassVar[dict[str,object]] = {
        "embebed_models":[ChromaDbClient]
    }
    # object attrs
    
    chroma_config:ChromaDbClient
    top_predictions:int = 3# number of predictions
    chunk_size:int = 2# extraer el id seleccionado al consultar - 1 para generar un par de mensajes
    current_collection:str = ""# the current table for default will be used the character_ia name
    path:str = "db/"# path to save the chomadb data
    embebingFunction:str = "all-MiniLM-L6-v2"# default for chomadb
    mode:str = "client"
    #allow_in_memory:bool = True

    

    def __post_init__(self):
        if(not(self.mode in ChromaDb.availableChromaDbModes)):
            raise NameError("error chomadb operation mode !")
        if(not(isdir(self.path))):
            raise NameError("DATABASE DOES NOT EXIST!")
    def __str__(self):
        ddata = dict((x,y) if not(isinstance(y,ChromaDbClient)) else (x,convert2Dict(y)) for x,y in vars(self).items() if not(x.startswith("_")))
        return dumps(ddata,indent=4)

@dataclass
class HyperDb:
    top_predictions:int = 3# number of predictions
    chunk_size:int = 2# extraer el id seleccionado al consultar - 1 para generar un par de mensajes
    current_collection:str = ""# the current table for default will be used the character_ia name
    path:str = "hyperdb/"# path to save the db
    rawFilesToIndex:str = join(path,"raw")
    pathDatabase:str = join(path,"db")
    embebingFunction:str = "all-MiniLM-L6-v2"# compatibility between chroma and hyperdb
    pathEmbebing:str = join(path,"embebbing")
    MAX_BATCH_SIZE:int = 2048 
    def __post_init__(self):
        if(not(isdir(self.path))):
            raise NameError("ERROR GLOBAL DATABASE FOLDER DOES NOT EXIST")
        if(not(isdir(self.pathDatabase))):
            raise NameError("DATABASE DOES NOT EXIST!")
        if(not(isdir(self.pathEmbebing))):
            raise NameError("EMBEBBING MODEL DOES NOT EXIST! ")
        
    def __str__(self):
        return convertObject2JsonData(self)
 


@dataclass
class properties:
    name:dict[str,str]
    age:dict[str,str]
@dataclass
class Function:
    name:str
    parameters:dict[str,str]
    required:list[str] 
@dataclass
class Tools:
    # clase para definir la estructura de las herramientas
    ''' 
        {
                "type": "function",
                "function": {
                  "name": "UserDetail",
                  "parameters": {
                    "type": "object",
                    "title": "UserDetail",
                    "properties": {
                      "name": {
                        "title": "Name",
                        "type": "string"
                      },
                      "age": {
                        "title": "Age",
                        "type": "integer"
                      }
                    },
                    "required": [ "name", "age" ]
                  }
                }
              }
    '''
    type:str
    function:Function
    tool_name:str
    tool_description:str
    tool_arguments:str
    
    def __post_init__(self):
        # resolve tool_filename to an function for call
        pass
    

ARGS_MOTIONS_DEFAULT = {
    'map_feelingExpressions':{
        'happy':'32'
    },
    'map_feelingModel':{
        'ride':'ride'
    },
    'increment_intimacy_sentiments':{
        'love':4,
        'amusement':2,
        'desire':5,
        'admiration':3,
        'excitement':5,
        'caring':5,
        'joy':2,
        'approval':2,
        'gratitude':4        
        },
    'decrement_intimacy_sentiments':{
        "anger":3,
        "annoyance":2,
        "grief":1,
        "disapproval":5,
        "disgust":4,
        "disappointment":5
    }
}
@dataclass
class MotionsInfo:
    _for='motions'
    # incrementar un nivel respecto a un key
    map_feelingExpressions:dict[str,str]
    # se usa para un mapa de sentimiento:expresion sentimiento:Motion
    
    map_feelingModel:dict[str,str]
    # dependiendo de alguna palabra clave se cambiara de un modelo a otro o desatara una expresion
    # por que usar este enfoque y no utilizar simplemente herramientas ?
    # por que este enfoque evitara que se realizen multiples llamadas al llm y ahorrara tiempo de reaccion
    increment_intimacy_sentiments:dict[str,str]
    decrement_intimacy_sentiments:dict[str,str]
    default_model:str = 'goth'
    default_expression:str = 'idle'
    def __str__(self):
        return convertObject2JsonData(self)

# MODELS AND EVENTS DESCRIPTION FOR INJECT TO THE PROMPT

@dataclass
class ModelDescription:
    # esta clase se usara para indicarle que hace cada modelo de live2d
    model_name:str# nombre del modelo ( no tiene nada que ver con el nombre del modelo que se usa en live2d para su ejecucion ) aqui debe ser un nombre descriptivo
    description:str
    _model_live2dId:str# id para ubicar el modelo y requerir su ejecucion al front
    expressions_description:dict[str,str]# = {"put an orgasm face":'expression01.exp'}
    model_motions_description:dict[str,str] #= {'hug':'hug the {user_prefix} face'}
    emotions_associed:list[str]# = ['love','desire']
    promptModel:str = ''# se arma en el post init
    def __post_init__(self):
        self.promptModel = f'{self.model_name} \n {self.description} \n The following are expressions that you can make:'
        self.promptModel += "\n".join([expression for expression in self.expressions_description.keys()])
        # expressions model
        self.promptModel += '\n and these are some moves you can do: '
        self.promptModel += "\n".join([motion for motion in self.model_motions_description.keys()])
    def __str__(self):
        return {self.model_name:{'description':self.description,'model_motions_description':self.model_motions_description}}
    
@dataclass
class EventPrompt:
    name:str# nombre del evento (ejemplo salir de cita)
    description:str# descripccion del evento ( ejemplo ir de cita con user_prefix)
    models:list[ModelDescription]
    # lista de modelos con descripcion y eso
    promptEvent:str = '*{ia_prefix} is in a pose {pose_description} *'
@dataclass
class levelPrompt:
    model:str
    level_number:int
    emotion:str
    prompt:str
    actionsPrompt:str
    actions:list[str]
    # events:dict[str,str]|EventPrompt los eventos seran disparados por el sentymental processor funcino
    # para ello se evaluara si se tiene el suficiente nivel de intimidad
    def toDict(self):
        return convert2Dict(self)
    

BASIC_LEVEL = {
    'nullLevel':levelPrompt(
        model='goth',
        level_number=226,
        emotion='distrust',
        prompt='{ia_prefix} considers {user_prefix} as a complete stranger {ia_prefix} will act modestly and with some distrust',
        actionsPrompt="From this moment {ia_prefix} can do the following actions:",
        actions=["smile"]
    ),
    'lowLevel':levelPrompt(
        model='goth',
        level_number=512,
        emotion='nervousness',
        prompt='{ia_prefix} barely knows {user_prefix} but she no longer considers him a complete stranger so she can begin to open up to him. ',
        actionsPrompt="From this moment {ia_prefix} can do the following actions:",
        actions=["smile"]
    ),
    'middleLevel':levelPrompt(
        model='goth',
        level_number=1024,
        emotion='caring',
        prompt='{ia_prefix} now considers a {user_prefix} as more than a close friend',
        actionsPrompt="From this moment {ia_prefix} can do the following actions:",
        actions=[
            "hug",
            "smile"
        ]
    ),
    'middleHightLevel':levelPrompt(
        model='goth',
        level_number=2048,
        emotion='caring',
        prompt='{ia_prefix} is starting to have feelings for {user_prefix}',
        actionsPrompt="From this moment {ia_prefix} can do the following actions:",
        actions=[
            "hug",
            "smile",
            'kiss',
            'blush',
            
        ]
    ),
    'hightLevel':levelPrompt(
        model='goth',
        level_number=4096,
        emotion='caring',
        prompt='{ia_prefix} now considers {user_prefix} as her boyfriend',
        actionsPrompt="From this moment {ia_prefix} can do the following actions:",
        actions=[
            "hug",
            "smile",
            'kiss',
            'sex',
            'blush',
            
        ]
    )
    
    
}
@dataclass
class IntimacyData:
    _for='intimacy'
    # levels of intimacy 
    nullLevel:levelPrompt|dict[str,str]
    lowLevel:levelPrompt|dict[str,str]
    # -----middle level-----
    middleLevel:levelPrompt|dict[str,str]
    middleHightLevel:levelPrompt|dict[str,str]
    # ---hight level---
    hightLevel:levelPrompt|dict[str,str]
    max_intimacyLevel:int = 4096
    def __post_init__(self):
        levels = [setattr(self,argument,getattr(self,argument).toDict()) if isinstance(getattr(self,argument),levelPrompt) else argument for argument in vars(self) if not(argument.startswith("_")) and (isinstance(getattr(self,argument),levelPrompt) or isinstance(getattr(self,argument),dict))]
        # convert the levels 2 a dictionary for serialization using json data
    def toDict(self):
        return convert2Dict(self)
    def get_levels(self):
        return [ level if isinstance(getattr(self,level),dict) else None for level in vars(self) if not(level.startswith("_")) ]
    def dict2Level(self):
        self.get_levels()
        [ setattr(self,level,levelPrompt(**getattr(self,level))) for level in levels if level != None ]
        print(type(self.hightLevel))
        # setattr(self,level,levelPrompt(**getattr(self,level)))
        # ahora convierte de diccionario a levelPrompt
        
    
    
@dataclass
class PromptDocument:
    """ use this class for make new prompts telling who the cyberwaifu are """
    
    _metha_info:ClassVar[dict[str,object]] = {
        "embebed_models":[MotionsInfo,IntimacyData]
    }
    context:str
    ia_prefix:str
    user_prefix:str
    text_example:str|list[dict[str,str]]
    personality:str
    motions:MotionsInfo|dict[str,str]# = MotionsInfo()# Not implemented yet
    intimacy:dict[str,str]|IntimacyData
    scenario:str = ""
    temp:float = 0.6
    intimacy_level:int = 0
    def __post_init__(self):
        if len(self.context) == 0:
            raise NameError("PROMPT ERROR: Context unexisting")
    def __str__(self):
        if isinstance(self.intimacy,IntimacyData):
            self.intimacy = self.intimacy.toDict()# convertimos a diccionario
        ddata = dict((x,y) if not(isinstance(y,MotionsInfo)) else (x,convert2Dict(y)) for x,y in vars(self).items() if not(x.startswith("_")))
        return dumps(ddata,indent=4)

@dataclass
class GenericPrompt:
    prompt:str
    model_configs:dict[str,int | str | list[str]] = None
    # model_configs es un diccionario con configuracione especial como top_p top_k temperature etc
    def __str__(self):
        return convertObject2JsonData(self)
  

@dataclass
class ChatBotSettings:
    
    '''
    in this settings you will be available to configure the following points:
        > use a vector db
        > what backend use
        > use summarization
    '''
    # class atributes
    available_backends:ClassVar[list[str]] = ["gpt4all","transformers","llamacpp","debug","llama_debug"]
    available_vectorStorageBackends:ClassVar[list[str]] = ["Chromadb","HyperDB",""]
    prompt_paths:ClassVar[str] = "prompt_paths/"
    # object attributes
    backend:str
    vectorStorageBackend:str
    chat_buffer_size:int# buffer for the conversation
    chat_format:str = 'pygmalion_dev'
    prompt_document:str = "ranni.json"# this will be used for load the prompt
    full_prompt_document:str = join(prompt_paths,prompt_document)
    prompt_summarization_document:str = "summarization.json"
    prompt_sentymental_analysis_document:str = "sentymental.json"
    prompt_memories_document:str = "memories.json"
    full_sentymental_analysis_document:str = join(prompt_paths,prompt_sentymental_analysis_document)
    full_summarization_document:str = join(prompt_paths,prompt_summarization_document)
    full_memories_document:str = join(prompt_paths,prompt_memories_document)
    use_vectorStoragedb:bool = False
    use_summarysation:bool = True
    use_sentymental_analysis:bool = False
    max_sumarization_lengt:int = 100
    min_sumarization_lengt:int = 20
    model_path:str = r"model/"# model/calypso-3b-alpha-v2
    # default configuration for the calypso model replace it with 
    # the other configuration of your model
    temperature:float = 0.6
    top_p:float = 0.75
    top_k:float=50
    num_beams:int = 50
    max_new_tokens:int = 256
    repetition_penalty:float = 1.4
    early_stopping:bool = True
    load_in_8bit:bool = False
    hook_storage:int =  0# numerod e mensajes para activar el almacenamiento # pode defecto la mitad del buffer size ( se saca la mitad en el post init)
    vector_storage_configuration_file:str = "chroma_db.json" # for default uses chomadb
    # specialized model for sentymental 
    specialized_sentymental_model:str = "specialized_sentymental_model"
    processEveryIaMessageAfterInference:bool = True
    # si procesara cada sentimiento de la ia y no esperara hasta que el hook de storage_hook 
    # para procesar todos los sentimientos (util al usar betalive para que tenga reacciones choerentes)
    use_specialized_model:bool = True
    # si es verdadero entonces no se usara el llm cargado se usara el modelo especializado
    # obly the backends with no _ at start will be converted to json data config
    
    def __post_init__(self):
        # claus guards
        if not(self.backend in ChatBotSettings.available_backends):
            raise NameError('Formatting Error ! \n unikown backend')
        if self.use_vectorStoragedb and not(self.vectorStorageBackend in ChatBotSettings.available_vectorStorageBackends):
            raise NameError('Formatting Error ! \n unikown database backend ')
        links = """
        https://huggingface.co/EleutherAI/gpt-neo-125M    ～500MB
        https://huggingface.co/EleutherAI/gpt-neo-1.3B   ～5GB
        https://huggingface.co/EleutherAI/gpt-neo-2.7B    ～10GB
        https://huggingface.co/EleutherAI/gpt-j-6B    ～12G(FP16) or ～24G(FP32)
        https://huggingface.co/EleutherAI/gpt-neox-20b    ～35GB!
        """
        # validates the existence of the model path
        if not(isdir(self.model_path)) or len(listdir(self.model_path)) == 0:
            if ( self.model_path.endswith('.ggml') and not(self.isfile(self.model_path))):    
                raise NameError(f'Unexisting model download one at the follow links: \n \t\t {links}')
        if not(isfile(join(ChatBotSettings.prompt_paths,self.prompt_document))):
            print(join(ChatBotSettings.prompt_paths,self.prompt_document))
            raise NameError("the Prompt document does not exists!")
        if(not(isfile(join('config',self.vector_storage_configuration_file)))):
            print(self.vector_storage_configuration_file)
            raise NameError("the Chomadb configuration file does not exists!")
        #self.full_prompt_document # this path will be used for access to the prompt_document and loadit
        
        if(self.hook_storage > self.chat_buffer_size):
            # si es mayor levantara un erro por que nunca se guardaran los datos
            raise NameError("Excetion invalid hook storage the value need to be lees than the buffer size")
        if(self.hook_storage == 0):
            self.hook_storage = self.chat_buffer_size//2
        if(not(all([isfile(x) for x in [self.full_prompt_document,self.full_sentymental_analysis_document,self.full_summarization_document]]))):
            raise NameError("Unexisting template file")
            ...
            
    def __str__(self):
        return convertObject2JsonData(self)


@dataclass
class Google_trans:
    _for = 'google_trans' # define en que llave se guardaran los datos
    trans_ipt:str = "zh"
    trans_opt:str = "zh"
    trans_opt2:str ="jpn"
    def __str__(self):
        return dict((x,y) for x,y in vars(self).items() if not(x.startswith("_")))
    def toDict(self):
        return dict((x,y) for x,y in vars(self).items() if not(x.startswith("_")))
@dataclass
class Baidu_trans:
    _for = 'baidu_trans'
    trans_ipt:str = "zh"
    trans_opt:str = "zh"
    trans_opt2:str = "jp"
    def __str__(self):
        return convertObject2JsonData(self)
    def toDict(self):
        return dict((x,y) for x,y in vars(self).items() if not(x.startswith("_")))


@dataclass
class Settings:
    ''' implementation of the original settings '''
    
    """ json code
    {
        {
        "useAccelerateModule":false # this will implements the accelerate module
        "conversation": true,
        "sentence": true,
        "no-lf": true,
        "no-prompt": true,
        "answer-only": false,
        "length": 12,
        "translator": "google",
        "google-trans": {
            "trans-ipt": "zh",
            "trans-opt": "zh",
            "trans-opt2": "jpn"
        },
        "baidu-trans": {
            "trans-ipt": "zh",
            "trans-opt": "zh",
            "trans-opt2": "jp"
        }
    }

    """ 
    # metha_info
    '''
    the metha info is information sued for know some aspects of the model
    like if it use embebed models like this one
     
    '''
    _metha_info:ClassVar[dict[str,object]] = {
        "embebed_models":[Google_trans,Baidu_trans]
    }
    # class Arguments
    availableTranslators:ClassVar[list[str]] = ["gooogle","baidu"]
    # Object Arguments
    google_trans:Google_trans
    baidu_trans:Baidu_trans
    useAccelerateModule:bool = False
    conversation:bool = True
    sentence:bool = True
    no_lf:bool = True
    no_prompt:bool = True
    answer_only:bool = False
    length:int = 12
    translator:str  = "google"
    def __post_init__(self):
        if not(self.translator in Settings.availableTranslators):
            raise ValueError('Unikown option of translation !')
        
    def __str__(self):
        ddata = dict((x,y) if not(isinstance(y,Google_trans) or isinstance(y,Baidu_trans)) else (x,convert2Dict(y)) for x,y in vars(self).items() if not(x.startswith("_")))
        return dumps(ddata,indent=4)




# print(ChatBotSettings(backend="gpt4all",vectorStorageBackend="Chomadb"))
'''
{
    "backend": "gpt4all",
    "vectorStorageBackend": "Chomadb",
    "use_vectorStoragedb": false,
    "use_summarysation": true
}
'''

# dataclass for save documents


@dataclass
class Metadata:
    _for = "metadatas"
    sumarization:str = ""
    sentimental_conversation:str = ""
    date:str = str(datetime.now())
@dataclass 
class Document:
    _metha_info:ClassVar[dict[str,object]] = {
        "embebed_models":[Metadata]
    }
    hyperdb_format:ClassVar[bool] = False
    sq_number:ClassVar[int] # use collection.count()
    metadatas:list[Metadata]
    documents:list[str]# ia prefix
    def __post_init__(self):
        
        if len(self.documents) == 1 or Document.hyperdb_format:
            #print("FF",len(self.documents))
            self.ids = Document.sq_number + 1
        else:
            #print("TT",len(self.documents),Document.sq_number)
            self.ids:list[str] = [str(x) for x in range(Document.sq_number+1,Document.sq_number + len(self.documents)+1)]# new uuid\
        # actualizamos numero de sequencia
        #print(self.ids)
        #self.ids = [ str(Document.sq_number) ]
        #Document.sq_number += int(self.ids[-1]) + 1# ultimo elemento 
        #self.metadatas = [ Metadata(x) for x in range(0,len(self.documents))]
        self.metadatas =  [ convert2Dict(x) if isinstance(x,Metadata) else x for x in self.metadatas]# convert embebed to dict
        # generates the id for each conversation
    def __str__(self):
        ddata = dict((x,y) if not(isinstance(y,Metadata)) else (x,convert2Dict(y)) for x,y in vars(self).items() if not(x.startswith("_")))
        return dumps(ddata,indent=4)
    def toDict(self):
        ddata = dict((x,y) if not(isinstance(y,Metadata)) else (x,convert2Dict(y)) for x,y in vars(self).items() if not(x.startswith("_")))
        return ddata
    

# DATABASE Handler

# from threading import th
from abc import ABC,abstractmethod

        
class BaseHandler(ABC):
    @abstractmethod
    def createDocument(self,doc,metha):
        pass
    @abstractmethod
    def handler(self):
        pass
    @abstractmethod
    def extractChunkFromDB(self,message):
        pass
    @abstractmethod
    def commit(self):
        ''' for save the transaction '''


class ChomaDBHandler(BaseHandler):
    def __init__(self,ia_prefix:str):
        self._client = None 
        self._ia_prefix = ia_prefix
        self._collection:chromadb.HttpClient|chromadb.PersistentClient = None
        with ModelLoader(configuration_name="bot_settings.json",ModelClass=ChatBotSettings) as ml:
            self._bot_config = ml
        with ModelLoader(configuration_name=self._bot_config.vector_storage_configuration_file,ModelClass=ChromaDb) as ml:
            self._chroma_config = ml# loads the file cpmfogiration 
    @property
    def client(self):
        ''' merges the client connection checks if the client was connected each call of the method '''
        if(self._client.heartbeat()):
            # if its connected
            return self._client
        else:
            raise NameError("choma connection error !")
    @client.deleter
    def client(self):
        ''' delete the database ''' 
        print("ALERT THIS ACTION WILL DELETE THE DATABASE !".center(30,"#"))
        self._client.reset()
    @property
    def collection(self):
        """ current collection loads """
        return self._collection
    @collection.setter
    def collection(self,collection_name:str):
        ''' load the colection ''' 
        self._collection = self.client.get_or_create_collection(collection_name)
    def loadOnLocalMode(self):
        self._client = chromadb.PersistentClient(path=self._chroma_config.path)
        # mas rapido que usar Settings
        # parquet default config and duckdb
    def runProcessOrHttpServer(self):
        from threading import Thread# imports the module
        t = Thread(target=self.loadOnClientProcessOrHttp)
        t.start()
    def loadOnClientProcessOrHttp(self):
        import subprocess
        cmd = ["chroma","run","--path",self._chroma_config.path]
        if self._chroma_config.chroma_config.host != 'localhost':
            cmd.insert(2,"--host")
            cmd.insert(3,self._chroma_config.chroma_config.host)
            print(cmd)
        r = subprocess.Popen(cmd)
        r.wait()
        # loads 
    def httpOrProcessClient(self):
        self._client = chromadb.HttpClient(port=self._chroma_config.chroma_config.port,host=self._chroma_config.chroma_config.host)
        
        # executes the server
    def extractChunkFromDB(self,message:str) ->list[str]|list:
        # funcion para extrar datos de la base de datos
        result = self._collection.query(
            query_texts=[message],# buscmos datos referentes al nuevo prompt e insertamos resultados
            n_results=self._chroma_config.top_predictions,
        )
        if len(result["documents"]) != 0 and not(result["documents"][0] == []): 
            # procedemos solo si si hubo un resultado
            print(result)
            message_id = int(result["ids"][0][0])# solo el primer id
            range_query = [str(x) for x in range(message_id,message_id + (self._chroma_config.chunk_size -1)) ]# restamos el mensaje que se uso para buscar el querty
            # volvemos a consultar desde la consulta que nos dio el id por complejidad
            chunk_response = self._collection.get(
                ids=range_query
            )
            
            return chunk_response
        return {}# diccionario vacio
            
    def createDocument(self,past_dialogue:list[str],metha:list[dict[str,str]]):
        """ this function will be called when the sumarization_hook has been hooked """
        # las conversaciones se guardaran cada self._bot_config.chat_buffer_size
        #print(past_dialogue)
        
        #print(type(str(past_dialogue)))
        Document.sq_number = self._collection.count()
        # fijamos el numero 
        dc = Document(
                documents=past_dialogue,
                       metadatas=metha).toDict()
        '''
    [Metadata(**metha)
                            sumarization="",
                            #TODO: solo se realizara la sumarizacion al primer elemento despues de superar el buffer y se aplicara a todos los elementos
                        ## en el bloque antes de superar el buffer
                        ## hola (sumarizacion: usuario saluda) ...  buffer size alcanzado  de nuevo realiza sumarizacion
                        ## TODO: metadatas eliminara
                        sentimental_conversation="happy"
        )] 
        '''
        print("PRE DC".center(30,"#"),dc)
        self.collection.add(**dc)
        ## TODO: COLOCAR DE DONDE A DONDE SE AGARRARA DE PAST_DIALOGUE COMO EN CODIGO ORIGINAL [-20:]
    def handler(self):
        """ this function will call the specific method what will load the chromadb in an specific mode """ 
        match self._chroma_config.mode:
            case "client":
                self.loadOnLocalMode()
            case "http_client" | "server_process":
                if self._chroma_config.chroma_config.only_http:
                    print("WARNING: you can install the lightwaight client only using \n\t >pip install chromadb-client")
                self.runProcessOrHttpServer()# runs the chomadb server
                self.httpOrProcessClient()# sets the client
            # "client","http_client","server_process"
        if len(self._chroma_config.current_collection) == 0:
            self.collection = self._ia_prefix# create o get the collection
            print(f"USING {self._ia_prefix} COLLECTION".center(20,"#"))
            return 0
        
        self.collection = self._chroma_config.current_collection
        print(f"USING {self._chroma_config.current_collection} COLLECTION".center(20,"#"))
        return 0 
    def commit(self):
        # not nidded for chroma
        pass
            # use the table 
        #self._client = chromadb
    #def EmbebingHuggingFaceModel(self):
    #    '''
    #    function embebing to use for create the collections in chomadb
    #    https://www.youtube.com/watch?v=RkYuH_K7Fx4&t=28s
    #    '''
    #    SentenceTransformer
        

if __name__ == '__main__':
    #print(getcwd())
    #print(Settings(
    #    google_trans=Google_trans(
    #        ),
    #    baidu_trans=Baidu_trans()
    #    ))
    #print(ChatBotSettings(backend='debug',vectorStorageBackend='Chomadb',chat_buffer_size=20))
    #
    #print(ChromaDb(chroma_config=ChromaDbClient()))
    #obj = ChatBotSettings(backend='transformers',vectorStorageBackend='Chomadb')
    #print(PromptDocument(context="hello world",ia_prefix="hehe"))
    #print(isfile(join(obj.prompt_paths,obj.prompt_document)))
    #print(join(obj.prompt_paths,obj.prompt_document))
    ##print(obj.m)
    #print(PromptDocument(context="you are ranni the witch and your frend {user} asked some questions:",ia_prefix="ranni",user_prefix="bk"))
    #with ModelLoader(configuration_name=obj.full_prompt_document,ModelClass=PromptDocument) as ml: 
    #    print(ml)
   
    #with ModelLoader(configuration_name="bot_settings.json",ModelClass=ChatBotSettings) as ml:
    #    print(ml)
    #    print(dir(ml))
    #    obj = ml
    #    print(ml.full_prompt_document)
    chroma = ChomaDBHandler(ia_prefix="ranni")
    chroma.handler()
    print(chroma.extractChunkFromDB(
        'hello'
    ))
    
    """
    #    print(ml.model_path)
    with ModelLoader(configuration_name=obj.full_prompt_document,ModelClass=PromptDocument,no_join_config_file_path=True) as ml: 
        print(ml)
    #print(Settings(google_trans=Google_trans(),baidu_trans=Baidu_trans()))
    #print(Google_trans())
    #with ModelLoader(configuration_name="test.json",ModelClass=Settings) as ml:
    #    print(dir(ml))
    #    print(ml.google_trans.trans_opt)
    
    chroma = ChomaDBHandler(ia_prefix="ranni")
    chroma.handler()
    # chroma.collection.delete("ranni")
    #Document.sq_number = chroma.collection.count()
    #print(Document.sq_number)
    #chroma.client.delete_collection(name="ranni")

    chroma.createDocument(
        past_dialogue=[
            'Caballero: Mi bella princesa, estáis bien? Espero que esa bestia no os haya hecho daño. \n Princesa: Estoy bien'
        ]
    )
    chroma.createDocument(  
            ['''princesa: Pero tú eres más que un simple servidor. Eres un héroe, un hombre noble y generoso. ¿Qué puedo hacer para recompensarte por tu gesta?
caballero: Nada, princesa. Tu sonrisa y tu felicidad son suficientes para mí.
princesa: ¿Seguro que no hay nada que desees? ¿Ni siquiera un beso?
caballero: Bueno, si insistes... Un beso sería un regalo maravilloso.
princesa: Entonces ven aquí y tómalo. (Se besan apasionadamente)''']
    #            
    )
    chroma.createDocument(
        ['''princesa: Estoy muy agradecida por tu valentía, caballero. Has arriesgado tu vida para salvarme de las garras del dragón.
            caballero: No hay de qué, princesa. Es mi deber y mi honor servir a la corona y proteger a la reina del reino.''']
    )

    #chroma.createDocument(  
    chroma.collection.add(
    documents=['''princesa: Estoy muy agradecida por tu valentía, caballero. Has arriesgado tu vida para salvarme de las garras del dragón.
            caballero: No hay de qué, princesa. Es mi deber y mi honor servir a la corona y proteger a la reina del reino.'''],
        metadatas=[{'ee':'cc'}],
        ids=[str(chroma._collection.count() + 1)])
    print(chroma.collection.get())
    rsp = chroma.collection.query(
        query_texts=[
            "caballero: recuerdas la primera vez que nos besamos ?"
        ]
    )
    print(len(rsp["documents"]))
    print(rsp["documents"][0])
    print(chroma.collection.count())
    print(chroma.collection.get()["documents"])           
    #print(chroma.collection.get(ids=["20"]))
    ##del(chroma.client)
    #
    #print(chroma._collection.query(
    #    query_texts="princesa"
    #))
    #print(chroma.collection.peek())
    #'''
    #print(Document(documents=["ranni: dear tainish \n tainish: yes my lady !"],
    #                   metadatas=Metadata(
    #    sumarization="",#TODO: solo se realizara la sumarizacion al primer elemento despues de superar el buffer y se aplicara a todos los elementos
    #    # en el bloque antes de superar el buffer
    #    # hola (sumarizacion: usuario saluda) ...  buffer size alcanzado  de nuevo realiza sumarizacion
    #    # TODO: metadatas eliminara
    #    sentimental_conversation="happy"
    #)).toDict())
    #'''
    #print(" \n" *100)
    #print("result",chroma.collection.query(
    #    n_results=2,
    #    query_texts=["ranni: recuerdas cuando fuimos juntos"]
    #))
    #print("ENDED")"""