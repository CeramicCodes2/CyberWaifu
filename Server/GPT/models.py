
from dataclasses import dataclass
from typing import ClassVar
from json import dumps,loads,JSONDecodeError
from os.path import join,isfile
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
    def __init__(self,configuration_name:str,ModelClass):
        '''
            configuration_name -> name of the config file
        
        '''
        self.settings_path = join(ModelLoader.settings_path,configuration_name)
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
            raise ValueError("File not found ! \n create the file !")
        self.file_raw = open(self.settings_path,'r')
        try:
            self.configs = loads(self.file_raw.read())
        except JSONDecodeError:
            raise NameError('BAD CONFIG FILE !')
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
class ChatBotSettings:
    
    '''
    in this settings you will be available to configure the following points:
        > use a vector db
        > what backend use
        > use summarization
    '''
    # class atributes
    available_backends:ClassVar[list[str]] = ["gpt4all","transformers"]
    available_vectorStorageBackends:ClassVar[list[str]] = ["Chomadb",""]
    # object attributes
    backend:str
    vectorStorageBackend:str
    use_vectorStoragedb:bool = False
    use_summarysation:bool = True
    # obly the backends with no _ at start will be converted to json data config
    
    def __post_init(self):
        # claus guards
        if not(self.backend in ChatBotSettings.available_backends):
            raise NameError('Formatting Error ! \n unikown backend')
        if self.use_vectorStoragedb and not(self.vectorStorageBackend in ChatBotSettings.available_vectorStorageBackends):
            raise NameError('Formatting Error ! \n unikown database backend ')
        
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
    conversation:bool = True
    sentence:bool = True
    no_lf:bool = True
    no_prompt:bool = True
    answer_only:bool = False
    length:int = 12
    translator:str  = "google"
    def __post_init(self):
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

if __name__ == '__main__':
    #print(getcwd())
    #with ModelLoader(configuration_name="bot_settings.json",ModelClass=ChatBotSettings) as ml:
    #    print(ml)
    #    print(dir(ml))
    #print(Settings(google_trans=Google_trans(),baidu_trans=Baidu_trans()))
    #print(Google_trans())
    with ModelLoader(configuration_name="test.json",ModelClass=Settings) as ml:
        print(dir(ml))
        print(ml.google_trans.trans_opt)