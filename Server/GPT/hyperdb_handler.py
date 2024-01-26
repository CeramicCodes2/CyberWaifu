from models import HyperDb,ModelLoader,Document,Metadata,BaseHandler,ChatBotSettings
from transformers import AutoTokenizer, AutoModel
from hyperdb import HyperDB
import torch
import torch.nn.functional as F
from os.path import isfile,join
from os import listdir
# temporal
from json import loads,dumps
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename='hyperdb.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class RawDocumentMerger:
    def __init__(self,operation:str):#,hyperdb_config:HyperDb):
        '''
        this class its used as a merger for the raw files
        
        '''
        self._operation:str = operation
        self._pathRawConfigs:str = "hyperdb/raw/"
        self._indexJson:str = join(self._pathRawConfigs,'index.json')
        #self.hyperdb_config:HyperDb = hyperdb_config
        self._indx:dict[str,str] = self.indexFile# gets the index file
        self._poolFiles:list = []
        self._poolFilesNames:list[str] = []
        self.noFilesChanges:bool = False
    
    def checkUnprocessedFiles(self) -> tuple[bool,list[str]]:
        
        files = listdir(self._pathRawConfigs)
        untracked = []
        lst = lambda x: False if untracked.append(x) else False 
        # funcion para incluir valores no trackeados
        return not(all([ True if x in self._indx else untracked.append(x) for x in files if not(x.startswith('index.json'))])),untracked
        # buscamos que todos los directorios no tengan archivos nuevos sin procesar
        # utilizamos un not para negar 
    @property
    def indexFile(self):
        # loads the index file 
        with open(self._indexJson,'r') as ml:
            lds = loads(ml.read())
        return lds
    @indexFile.setter
    def indexFile(self,indexJson:dict[str,str]):
        '''update the index file'''
        with open(self._indexJson,'w') as ml:
            ml.write(dumps(indexJson))
    
    @property
    def poolFiles(self):
        ''' return the pool of files'''
        return self._poolFiles
    @poolFiles.setter
    def poolFiles(self,file):
        self._poolFiles.append(file)
    @poolFiles.deleter
    def poolFiles(self):
        for x in self._poolFiles:
            x.close()
    @property
    def poolFilesNames(self) ->list[str]:
        return self._poolFilesNames
    @poolFilesNames.setter
    def poolFilesNames(self,arg):
        self._poolFilesNames.append(arg)
    @poolFilesNames.deleter
    def poolFilesNames(self):
        self._poolFilesNames:list[str] = []
    def updateIndex(self,file:str,name_saved:str):
        ''' 
        this function will be called when the
        file was processed
        and will be used for insert new values
        
        '''
        if isfile(self._indexJson):
            self._indx[file] = name_saved
    def commit(self):
        ''' function for update de intex file json ''' 
        self.indexFile = self._indx
    def __enter__(self):
        status,files = self.checkUnprocessedFiles()
        logging.warn(status)
        if status:
            for file in files:
                self.poolFilesNames = file
                logging.info(msg=f'loadding the file {file}')
                self.poolFiles = open(join(self._pathRawConfigs,file),self._operation)
            self.noFilesChanges = True
            
        return self
    def __exit__( self, exc_type, exc_val, exc_tb ):
        del self.poolFiles
        # cerramos archivos
        # hacemos commit

class HyperDBPlus(HyperDB):
    # implementacion de mas acciones que hyperdb por defecto no implementa
    def __init__(self,
        hyperdb_configs:HyperDb,
        documents=None,
        vectors=None,
        key=None,
        embedding_function=None,
        similarity_metric="cosine",
    ):
        
        super().__init__(documents,vectors,key,embedding_function,similarity_metric)
        self._collection_path:str = None
        self.hyperdb_configs = hyperdb_configs
        #self._doc_keys = self.
    def count(self) -> int:
        ''' count operation return the number of registers existing in the db'''
        return len(self.documents) -1 # contamos desde cero
    def get(self,key:int|list[int]) -> dict[str,str]:
        '''return the spected register using a key index
        using a index 
        '''
        if isinstance(key,list):
            elements = []
            for item in key:
                try:
                    logging.info(self.documents[item])
                    elements.append(self.documents[item])
                except: 
                    logging.error(f'ERROR ELEMENT ERROR {item}')
            return elements
        return self.documents[key]#resolve the number key  
    def create_collection(self,collection_name:str,meta:list[dict[str,str]]=None):
        #self.__init__()
        logging.info('creating void db ...')
        logging.info(Metadata())
        Document.sq_number = 0
        dc = Document(
            documents=[''],
            metadatas=[meta or Metadata()]).toDict()
        self.add(dc)
        self._collection_path = collection_name
        self.save(collection_name)
            
    def get_or_create_collection(self,collection_name:str):
        fpath = join(self.hyperdb_configs.pathDatabase,collection_name)
        if isfile(fpath):
            logging.info(f'loadding ... {fpath}')
            self.load(fpath)
        else:
            self.create_collection(fpath)
        self._collection_path = fpath
        return self
    def peek(self):
        ''' 
        return all registers in the database
        '''
        return self.documents
        
    def commit(self):
        # method used for save the changes
        self.save(self._collection_path)
        
             
class HyperDBHandler(BaseHandler):
    def __init__(self,ia_prefix:str):
        self._client = None
        self._ia_prefix = ia_prefix
        with ModelLoader(configuration_name="bot_settings_cpy.json",ModelClass=ChatBotSettings) as ml:
            self._bot_config = ml
        with ModelLoader(configuration_name=self._bot_config.vector_storage_configuration_file,ModelClass=HyperDb) as ml:
            self._hyperdb_config = ml# loads the file cpmfogiration 
        self.MAX_BATCH_SIZE = self._hyperdb_config.MAX_BATCH_SIZE#2048 DEFAULT BATCH 
        self._collection = None
        ml = join(self._hyperdb_config.pathEmbebing,self._hyperdb_config.embebingFunction)
        self._tokenizer = AutoTokenizer.from_pretrained(ml)
        self._model = AutoModel.from_pretrained(ml)
    def __del__(self):
        # guardamos datos
        # self.collection.commit()
        ...
    def extractChunkFromDB(self,message:str) -> list[str]|list:
        MAX_DATABASE_REGISTERS = self.collection.count() # utilizamos este valor para evitar utilizar valores que no existen si se consulta un valor que esta al final de nuestro data set 
        result:list[tuple[dict[str,str]]] = self._collection.query(
            query_text=message,
            top_k=self._hyperdb_config.top_predictions
        )
        # logging.info(f'contente querty response {result} {not(result[0][0] == ({},))}')
        if len(result) != 0:
            message_id,distance = result[0]#['ids']
            message_id = message_id['ids']
            logging.info(message_id)
            range_query:list[int] = [x for x in range(message_id,message_id + (self._hyperdb_config.chunk_size )) if x <= MAX_DATABASE_REGISTERS]
            
            logging.error(f'index: {range_query}')
            
            chunk_response = self._collection.get(
            key=range_query
            )
            # hace falta convertir a un objeto como chromadb esdecir 
            #haremos uso de Document para ello 
            if chunk_response == []:
                return {}
            Document.sq_number = chunk_response[0]['ids']
            methas = []
            docs = []
            [(docs.extend(doc['documents']),methas.extend(doc['metadatas'])) for doc in chunk_response]
            logging.info('DOC METHAS')
            logging.info(methas)
            dc = Document(
                documents=docs,
                metadatas=methas
            ).toDict()
            logging.error('DATA DICT')
            logging.error(dc)
            return dc
        return {}
    def mean_pooling(self,model_output, attention_mask):
        '''
                https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        
        obtenido de 
        '''
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def encodedInputProcess(self,sentences):
        ''' 
        https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        
        obtenido de 
        '''
        encoded_input = self._tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self._model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings    
    def embebbing(self,documents:list[dict[str,str]],key=None):
        ''' function used for load the embebbing model 
        
        '''
        texts = []
        if isinstance(documents, list):
            #procesamiento de la clave
            if isinstance(documents[0], dict):
                # leemos el primer registro
                if isinstance(key, str):
                    if "." in key:
                        key_chain = key.split(".")
                    else:
                        key_chain = [key]
                    for doc in documents:
                        for key in key_chain:
                            doc = doc[key]
                        texts.append(doc.replace("\n", " "))
                elif key is None:
                    for doc in documents:
                        text = ", ".join([f"{key}: {value}" for key, value in doc.items()])
                        texts.append(text)
            elif isinstance(documents[0], str):
                texts = documents
        batches = [
            texts[i : i + self.MAX_BATCH_SIZE] for i in range(0, len(texts), self.MAX_BATCH_SIZE)
        ]
        # cut the document in batches
        embeddings = []# embebbings dict
        for batch in batches:
            # process every batch
            response =  self.encodedInputProcess(batch)#openai.Embedding.create(input=batch, model=model)
            embeddings.extend(response)#np.array(item["embedding"]) for item in response["data"])
        return embeddings 
    @property
    def collection(self) -> HyperDBPlus:
        
        return self._collection
    @collection.setter
    def collection(self,collection_name:str) -> HyperDBPlus:
        #fcoll =join(self._hyperdb_config.pathDatabase,self._ia_prefix + 'pickle.gz')
        if not(isfile(join(self._hyperdb_config.pathDatabase,collection_name))):
            # si el archivo no existe
            logging.warn(f"ALERT COLLECTION {collection_name} DOES NOT EXISTS CREATING ...")
        self._collection = HyperDBPlus(embedding_function=self.embebbing,
                                       hyperdb_configs=self._hyperdb_config,
                                       key = 'documents').get_or_create_collection(collection_name)
        # self._collection = collection_name     
    def handler(self):
        #print(self._hyperdb_config.pathEmbebing)
        if len(self._hyperdb_config.current_collection) == 0:
            self.collection = self._ia_prefix + '.pickle'# create o get the collection
            
            print(f"USING {self._ia_prefix} COLLECTION".center(20,"#"))
            return 0
        
        self.collection = self._hyperdb_config.current_collection
        print(f"USING {self._hyperdb_config.current_collection} COLLECTION".center(20,"#"))
            
    def indexer(self):
        ''' 
        this function its used for 
        process the data from a file like the format:
        {
            "name":"hello",
            "id":1
        }
        and parse to a picke.gz file

        '''
        logging.warning('start indexer')
        with RawDocumentMerger(operation='r') as rwh:
            #print(rwh.noFilesChanges)
            if rwh.noFilesChanges:
                # si hay cambios
                # logging.error(filename,file)
                logging.error('files changes detected !')
                for file,filename in zip(rwh.poolFiles,rwh.poolFilesNames):
                    documents = []
                    for line in file.readlines():
                        documents.append(loads(line))
                        logging.warning(line)
                    logging.info(documents)
                    instance = HyperDBPlus(hyperdb_configs=self._hyperdb_config,documents=documents,key='ids',embedding_function=self.embebbing)
                    logging.warn(f'instance loaded {filename}')
                    instance.save(join(self._hyperdb_config.pathDatabase,filename.split('.')[0] + '.pickle'))
                    rwh.updateIndex(file=filename,name_saved=join(self._hyperdb_config.pathDatabase,filename.split('.')[0] + '.pickle'))
                    
                rwh.commit()
                    
                        
                    #self.embebbing(file.read(),key='id')# leemos
                    
            # si no hay cambios
            
        return 0

    def createDocument(self,past_dialogue:list[str],metha:list[dict[str,str]]):
        
        Document.sq_number = self._collection.count()
        Document.hyperdb_format = True
        metha = metha if isinstance(metha,list) else [metha]
        dc = Document(
                documents=past_dialogue,
                       metadatas=metha
                       ).toDict()
        #dc['ids'] = int(dc['ids'][0])
        print("PRE DC".center(30,"#"),dc)
        logging.info(dc)
        self.collection.add(dc)
    def commit(self):
        self.collection.commit()
        
if __name__ == '__main__':
    #print(HyperDb(current_collection='ranni'))
    
    client = HyperDBHandler(ia_prefix='ranni')
    #client.indexer()
    client.handler()
    #print(client.collection.peek())
    #
    #print(client.collection.get(3))
    #client.createDocument(
    #    past_dialogue=["ORDER: IRQ RANNI"],
    #    metha=Metadata()
    #)
    #client.commit()
    #del client
    #print(client.collection.peek())
    print(client.extractChunkFromDB(
        message='jcka'
    ))
    #print(client.collection.peek())
    #client.commit()
    #client.createDocument(
    #    past_dialogue=["ORDER: IRQ RANNI"],
    #    metha=Metadata()
    #)
    '''
    client = HyperDBHandler(ia_prefix='ranni')
    client.indexer()
    client.handler()
    #print(client.collection.count())
    

    #res = client.collection.query(
    #    'IRQ',
    #    top_k=5
    #)
    client.collection.commit()
    '''
    
    #print(client.collection.peek())
    #print(res)
    #from pickle import loads
    #op = open('./hyperdb/db/ranni.pickle','rb')
    #print(loads(op.read()))
    #print(client.collection.get(0))
    #client.createDocument(
    #    past_dialogue=['ranni:hello blake'],
    #    metha={"summary":'hello'}
    #    
    #)
    #Document.sq_number = 1
    #dc=  Document(
    #    documents=[
    #        "blake: hello ranni",
    #        "ranni: hi blake"
    #        ],
    #    metadatas=[
    #        Metadata()
    #    ]
    #).toDict()
    #print(dc)