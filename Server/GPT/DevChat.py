#import gradio as gr
#import torch
from models import ModelLoader,ChatBotSettings,Settings,Google_trans,Baidu_trans,PromptDocument,join,Metadata,GenericPrompt
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename='logging.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

TRANSFORMER_CHAT_FORMAT = {}

def register_transformer_chat(template_name:str):
    def callback(function):
        TRANSFORMER_CHAT_FORMAT[template_name] = function
        return function
    return callback

@register_transformer_chat("gpt_neo")
def gpt_neo(user_alias,character_name,personality,text_example,conv_prompt,past_dialogue_formatted,message):
    return f"""
Let's do a role-playing game
\n
You must pretend that you are {character_name}
The following text describes {character_name}'s personality:

{conv_prompt}
you must follow the following rules:
- you must generate a response for the user {user_alias} but stop generating a response as soon as
  you are speaking for the user.
- Not to mention that he is a model of artificial intelligence.
- Use a seductive and enigmatic tone of voice.
- Respond to my questions and comments, but also ask me questions about my case.

The following texts are an example of something you would say
{text_example}

Reply to the messages made by {user_alias} according to your personality.

This is the conversation between {user_alias} and {character_name} till now:
{past_dialogue_formatted}

Continuing from the previous conversation, write what {character_name} says to {user_alias}:
{user_alias}: {message}
{character_name}:"""
#from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Protocol
#import llama_cpp.llama_types as llama_types
#from llama_cpp.llama_chat_format import register_chat_format,ChatFormatterResponse,_get_system_message,_map_roles,_format_chatml

class Chat:
    def __init__(self, message_history=[],inject_chat_prompt=True):
        self.load_settings()
        self.conv_prompt = self._prompt_document.context# .format(user=self._prompt_document.user_prefix)# loads the context of the situation
        self.inject_chat_prompt = inject_chat_prompt
        self.user_alias = self._prompt_document.user_prefix
        self.character_name = self._prompt_document.ia_prefix
        self.chat_buffer_size = self._chatSettings.chat_buffer_size
        self.message_history = message_history
        self.display_messages = []# mensajes a mostrar
        self.storage_hook = self.message_history.copy()
        self.pairRegister2Block = lambda x,y: [f"{x['role']}: {x['content']}",f"{y['role']}: {y['content']}"]
        self._prompt = ''
        self._memories_message = ''
        self.use_llama = False
        self._chat_injection_Prompt = False# use cuando ya se haya inyectado el prompt en el caso de usar create_chat
        # para asegurarse de solo injectar el prompt una vez
        self.generator:pipeline|Llama|None = None
        #self._database = None
        for message_pairs in message_history:
            message1, message2 = message_pairs
            self.display_messages.append([message1['content'], message2['content']])
        self.loadModel()
    @property
    def database(self):
        return self._database
    @database.setter
    def database(self,arg:list[str]):
        self._database.createDocument(arg)
        # save data
    def transformer_backend(self):
        from transformers import pipeline,GenerationConfig 
        generation_config = GenerationConfig(
            temperature=self._chatSettings.temperature,
            top_p=self._chatSettings.top_p,
            top_k=self._chatSettings.top_k,
            do_sample=True,
            num_beams = self._chatSettings.num_beams,
            early_stopping=self._chatSettings.early_stopping,
            repetition_penalty=self._chatSettings.repetition_penalty,
            #temperature=temperature,
            #top_p=top_p,
            #top_k=top_k,
            #num_beams=num_beams,
            #early_stopping=True,
            #repetition_penalty=repetition_penalty,
        )
        #model = LlamaForCausalLM.from_pretrained(model=self._chatSettings.model_path,config=GenerationConfig,load_in_8bit=self._chatSettings.load_in_8bit)
        #tokenizer = LlamaTokenizer.from_pretrained(model=self._cha)
        self.generator = pipeline('text-generation',
                                  do_sample=True,
                                  model=self._chatSettings.model_path,
                                  device_map="auto",
                                  max_length=self._chatSettings.max_new_tokens,
                                  )
        self.conv_analysis = lambda input_data: self.generator(input_data,task="sentiment-analysis")
        #if self._chatSettings.use_summarysation:
        #    s#elf.summarizator = lambda input_data: self.generator(input_data,task="summarization",min_length=5, max_length=20)#max_length=self._chatSettings.max_sumarization_lengt)       
    def llamaCpp_backend(self):
        from llama_cpp import Llama,llama_tokenize
        from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Protocol
        import llama_cpp.llama_types as llama_types
        from llama_cpp.llama_chat_format import register_chat_format,ChatFormatterResponse,_get_system_message,_map_roles,_format_chatml
        @register_chat_format("pygmalion_dev")
        def format_pygmalion_dev(
            messages: List[llama_types.ChatCompletionRequestMessage],
            **kwargs: Any,
        ) -> ChatFormatterResponse:
            system_template = """<|system|>{system_message}""" 
            # """<|system|>Enter RP mode. Pretend to be {ia_prefix} whose persona follows: \n {system_message} \n You shall reply to the user while staying in character, and generate long responses. \n <START> \n"""
            system_message = _get_system_message(messages)
            system_message = system_template.format(system_message=system_message,ia_prefix=self._prompt_document.ia_prefix)
            
            
            _roles = {self._prompt_document.user_prefix:"<|user|>",
                      self._prompt_document.ia_prefix:"<|model|>"
                      } #dict(blake="<|user|>", ranni="<|model|>")
            _sep = "\n"
            _messages = _map_roles(messages, _roles)
            _messages.append((_roles[self._prompt_document.ia_prefix], None))
            _prompt = _format_chatml(system_message, _messages, _sep)
            logging.info('CHAT FORMAT INFO'.center(50,'-'))
            logging.info('messages')
            logging.warn(_messages)
            logging.info('prompt')
            logging.error(_prompt)
            
            logging.info("END OF CHAT FORMAT INFO".center(50,'-'))
            return ChatFormatterResponse(prompt=_prompt, stop=_sep)
        self.ugenerator = Llama(model_path=self._chatSettings.model_path,
                               chat_format=self._chatSettings.chat_format,#"pygmalion_dev",
                               max_tokens=self._chatSettings.chat_buffer_size,
                               n_gpu_layers=-1,
                               main_gpu=0,
                               n_ctx=1024
                               )
        
        self.text_completation = lambda prompt,option: self.ugenerator(prompt=prompt,**self._summarizator_model_configs) if option else self.ugenerator(prompt=prompt,**self._sentymental_model_configs)
        # True -> summarizar False -> sentimental configs
        #self.cg = self.ugenerator.create_chat_completion(messages,temperature=self._chatSettings.temperature,top_p=self._chatSettings.top_p,top_k=self._chatSettings.top_k)
        
        self.generator = lambda messages: self.ugenerator.create_chat_completion(messages,temperature=self._chatSettings.temperature,top_p=self._chatSettings.top_p,top_k=self._chatSettings.top_k)
    def loadModel(self):
        match self._chatSettings.backend:
            case "transformers":
                self.transformer_backend()
                self.evaluate = self.transformers_evaluate
            case "llamacpp":
                self.use_llama = True
                self.llamaCpp_backend()
                self.evaluate = self.llama_evaluate
            case "gpt4all":
                pass
            case "debug":
                self.generator =  lambda text: [{"generated_text":text}]
            case "llama_debug":
                solve = {
                            "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                            "object": "text_completion",
                            "created": 1679561337,
                            "model": "./models/7B/llama-model.gguf",
                            "choices": [
                              {
                                "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": "stop"
                              }
                            ],
                            "usage": {
                              "prompt_tokens": 14,
                              "completion_tokens": 28,
                              "total_tokens": 42
                            }
                }
                chat_solve = {'id': 'chatcmpl-dfd730c5-e868-43b0-bc89-f76e3a1848fa', 'object': 'chat.completion', 'created': 1705535043, 'model': 'model/mistral-pygmalion-7b.Q4_K_M.gguf', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 421, 'completion_tokens': 0, 'total_tokens': 421}}
                
                self.use_llama = True
                self.ugenerator = lambda prompt,**kwargs: solve
                self.text_completation = lambda prompt,option: self.ugenerator(prompt=prompt,**self._summarizator_model_configs) if option else self.ugenerator(prompt=prompt,**self._sentymental_model_configs)
                self.generator = lambda messages: chat_solve
                self.evaluate = self.llama_evaluate
            
        
        self.resolve_backend = lambda pre_summary,use_llama,option: self.text_completation(prompt=pre_summary,option=option) if use_llama else self.generator(pre_summary)      
    def load_settings(self):
        with ModelLoader("bot_settings.json",ChatBotSettings) as model:
            self._chatSettings = model
        with ModelLoader(configuration_name=self._chatSettings.full_prompt_document,ModelClass=PromptDocument,no_join_config_file_path=True) as ml:
            self._prompt_document = ml
            self.character_name = ml.ia_prefix
            self.user_alias = ml.user_prefix
            # NOTE: if a temp was defined it will replace the default configuration temperature
            self._chatSettings.temperature = ml.temp
        
        #self._prompt_sentymental = ''
        with ModelLoader(configuration_name=self._chatSettings.full_sentymental_analysis_document,ModelClass=GenericPrompt,no_join_config_file_path=True) as ml:
            self._sentymental_model_configs = ml.model_configs
            
            self._prompt_sentymental = ml.prompt
        with ModelLoader(configuration_name=self._chatSettings.full_summarization_document,ModelClass=GenericPrompt,no_join_config_file_path=True) as ml:
            self._summarizator_model_configs = ml.model_configs
            self._prompt_summarizator = ml.prompt
        with ModelLoader(configuration_name=self._chatSettings.full_memories_document,ModelClass=GenericPrompt,no_join_config_file_path=True) as ml:
            self._prompt_memories = ml.prompt
            
        # INIT DATABASE
        handler = None
        if self._chatSettings.vectorStorageBackend == 'Chromadb':
            from models import ChomaDBHandler,ChromaDb,ChromaDbClient
            handler = ChomaDBHandler(ia_prefix=self._prompt_document.ia_prefix)
        elif self._chatSettings.vectorStorageBackend == 'HyperDB':
            from hyperdb_handler import HyperDBHandler
            handler = HyperDBHandler(ia_prefix=self._prompt_document.ia_prefix)
            
        self._database = handler
        self._database.handler()
    
    def gpt_neo(self,prompt):
        return self.generator(prompt, do_sample=True, min_length=20, max_length=self._chatSettings.max_new_tokens)
    @property
    def get_prompt(self):
        return self._prompt
    @get_prompt.setter
    def get_prompt(self,arg):
        self._prompt = arg
    def evaluate(self, message, **kwargs):#  temperature=0.6, top_p=0.75, top_k=50, num_beams=5, max_new_tokens=256, repetition_penalty=1.4,
        pass
        #self._chatSettings.backend == 'transformers':
        #    self.transformers_evaluate(message, **kwargs)
        
    def llama_evaluate(self,message,**kwargs):
        prompt = self.llama_propt_gen_chat(self.message_history,message)
        # print(prompt)
        return prompt
        
    def transformers_evaluate(self, message, **kwargs):
        prompt = self.prompt_gen_chat(self.message_history, message)
        #self.get_prompt = prompt
        if self.use_llama:
            output = self.generator(prompt)["choices"][0]
        else:
            output = self.generator(prompt)[0]["generated_text"]
        
        # output = self.gpt_neo(prompt=prompt)[0]["generated_text"]
        # generator = pipeline('conversational', model=r'model/gpt-neo-125m')
        #split_str = f"""### Response:\n{self.character_name}:"""
        #output = output.split(split_str)[1].strip()
        return output
    def reset_message_history(self,message,response,save_overflow_conversation=True):
        ''' 
        argument save_overflow_conversation will save the conversation using a vector storage database
        {
         "sumarization":"",
         "messages":{
             
         }   
        }
        '''
        if(len(self.message_history) > self.chat_buffer_size):
            #self.convert2Blocks(self.message_history[:-self.chat_buffer_size])
            self.message_history = self.message_history[-self.chat_buffer_size:]
            # toma del ulimo elemento en adelante
        if(len(self.storage_hook)%self._chatSettings.hook_storage == 0):
            # pregunta respuesta
            #print('PRCCC')
            doc = []
            metha = []
            #print(self.storage_hook)
            self.storage_hook = self.storage_hook[-self._chatSettings.hook_storage:]# recorremos
            #print("CALLING SUMMARIZATOR")
            logging.info('CALLING SUMMARIZATOR')
            
            self.summarizator(use_llama=self.use_llama)
            #self.sentimental_analysis(self.storage_hook)
            logging.info('ACKA')
            logging.info(self.storage_hook)
            
            [ [metha.extend([x.get("methadata",False),y.get("methadata",False)]),doc.extend([f"{x['role']}: {x['content']}",f"{y['role']}: {y['content']}"])] for x,y in self.storage_hook]
            # sumarizamos
            #print(doc)
            logging.warn(doc)
            
            
            #[ sh.extend([print(x,y)]) for x,y in self.storage_hook]
            metha = [ x if x else Metadata(date=str(datetime.now())) for x in metha]
            # limpiamos datos para guardarlos
            #print(doc)#self.storage_hook[:self._chatSettings.hook_storage])
            #logging.info("SENTYMENTAL CALL".center(50,"#"))
            #self.sentymental()
               
            self._database.createDocument(doc,metha=metha)#[:self._chatSettings.hook_storage]))
            # al activarse se guardan los primeros elementos
            self._database.commit()# makes the commit
            #print("HOOK".center(50,"#") + '\n',self.storage_hook)
            #print("HOOK".center(100,"#") + "\n",self.storage_hook)
        #self.storage_hook.append([message,response])
        self.display_messages.append([message, response])
        return self.display_messages
        #return self.display_messages
    def sentimental_analysis(self,conversation):
        for x,y in conversation:
            #print(self.pairRegister2Block(x,y))
            lysis = self.conv_analysis(self.pairRegister2Block(x,y))
            conversation["methadata"]["sentimental_conversation"] = lysis[0]
            conversation["methadata"]["sentimental_conversation"] = lysis[1]
            # https://huggingface.co/docs/transformers/v4.36.1/en/quicktour#pipeline
            
    def update_conversation(self,message:str):
        ''' update the conversation add and format the messages ''' 
        response = self.evaluate(message)
        if self.use_llama:
            response = response["choices"][0]["message"]["content"]
        logging.error(response)
        self.message_history.append(
            (
                {"role": self.user_alias, "content": message},
                {"role": self.character_name, "content": response},
            )
        )
        logging.info("STATUS APPEND".center(50,"="))
        logging.warning(response)
        logging.warning(message)
        self.storage_hook.append(
            (
                {"role": self.user_alias, "content": message,"methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}},
                {"role": self.character_name, "content": response,"methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}},
            )
        )# usar el metodo copy demandaria mayor gasto de recursos
        #self.message_history.copy()
        # append the new message
        
        display = self.reset_message_history(message,response)
        return display
    def mapReduceSummarizator(self):
        ''' not implemented yet'''
        pass
    def summarizator(self,use_llama:bool):
        ''' stuff summarization requieres an llm https://python.langchain.com/docs/use_cases/summarization'''
        
        # guardar con todo el historial o solo los bloques especificos  ?
        # por ahora solo los  bloques especificados
        gblock = []
        for x,y in self.storage_hook:
            gblock.extend(self.pairRegister2Block(x,y))
        gblock = ' \n'.join(gblock)
        pre_summary = self._prompt_summarizator.format(messages=gblock,ia_prefix=self._prompt_document.ia_prefix,user_prefix=self._prompt_document.user_prefix)
        logging.error(pre_summary)
        #resolve_backend = lambda use_llama,option: self.text_completation(prompt=pre_summary,option=option) if use_llama else self.generator(pre_summary)
        summary = self.resolve_backend(pre_summary=pre_summary,use_llama=use_llama,option=True)
        logging.info("SUMARY".center(50,"="))
        logging.error(summary)
        #print(summary)
        if use_llama:
            summary = summary["choices"][0]["text"]
        else:
            summary = summary[0]["generated_text"]
        logging.error(summary)
        #gblock = ' \n'.join([ self.pairRegister2Block(x,y) for x,y in block]) 
        # TODO GUARDAR EN EL self.storage_hook
        for idx,item in enumerate(self.storage_hook):
            x,y = item
            x['methadata']['sumarization'] = summary
            y['methadata']['sumarization'] = summary
            item = (x,y)
            #print("summary",item)
            self.storage_hook[idx] = item # dict
        #block[]
        #print(self.storage_hook)
        
        pass
    def sentymental(self):
        '''
        sentymental analysis
         
        '''
        for x,y in self.storage_hook:
            # gblock.extend(self.pairRegister2Block(x,y))
            pre_prompt = self._prompt_sentymental.format(message=self.pairRegister2Block(x,y),ia_prefix=self._prompt_document.ia_prefix,user_prefix=self._prompt_document.user_prefix)
            gen = self.resolve_backend(use_llama=self.use_llama,option=False)
            gen = self.generator(pre_prompt)[0]["generated_text"]
            x['methadata'][f'sentimental_conversation'] = gen
            y['methadata'][f'sentimental_conversation'] = gen
        #gblock = ' \n'.join([ self.pairRegister2Block(x,y) for x,y in block]) 
        # TODO GUARDAR EN EL self.storage_hook        
    @property
    def databasec(self):
        return self._database
    def llama_injectExamples(self,main_dct):
        logging.info("TEXT EXAMPLE".center(50,"-"))
        logging.warn(type(self._prompt_document.text_example))
        logging.info(self._prompt_document.text_example)
        if not isinstance(self._prompt_document.text_example,str):
            logging.warn('USING NOT STR OPTION'.center(10,'-'))
            main_dct.append({"role":"system","content":"The following are comments that Ranni would say:"})
            main_dct.extend(self._prompt_document.text_example)
        else:
            main_dct.append({"role":"system","content":f"The following are comments that Ranni would say: \n {self._prompt_document.text_example}"})
    def llama_propt_gen_chat(self,message_history,message):
        '''
        como es un chat se debe de injectar el prompt al inicio 
         
        '''
        main_dct = []
        # prompt parainyectar recurdos
        promptMemories = self._prompt_memories
        if self._chat_injection_Prompt:
            logging.warn(message_history)
            main_dct.extend(self.pair2tuple(message_history))
            main_dct.append({"role":self.user_alias,"content":f"{message}"})
            self.inject_llama_memories(main_dct,message,promptMemories)
            logging.error(main_dct)
            return self.generator(
                messages=main_dct
            )
        prp = f"""Enter RP mode. Pretend to be {self.character_name} whose persona follows: \n {self.conv_prompt}\n{self._prompt_document.personality} \n world scenario: {self._prompt_document.scenario} \n You shall reply to the user while staying in character, and generate long responses. \n <START> \n" 
        """
        sysHist = f'''\n This is the conversation between {self.user_alias} and {self.character_name} till now: \n'''        
        usrIndication = f"\n You shall reply to the user while staying in character, and generate long responses. \n <START> \n"
        usrInput = f"{message}\n"
        main_dct.append({"role":"system","content":prp})
        self.llama_injectExamples(main_dct)
        main_dct.append({"role":"system","content":sysHist})
        main_dct.extend(self.pair2tuple(message_history))
        main_dct.append({"role":"system","content":usrIndication})
        main_dct.append({"role":self.user_alias,"content":usrInput})
        logging.info(main_dct)
        self.get_prompt = main_dct
        #print(main_dct)
        self._chat_injection_Prompt = True
        #logging.error(main_dct)
        return self.generator(
            messages=main_dct
        )
    def pair2tuple(self,messages:tuple[dict[str,str]]) -> list[dict[str,str]]:
        lstdct:tuple[dict[str,str]] = []
        for pairs in messages:
            x,y  = pairs
            lstdct.append(x)
            lstdct.append(y)
        return lstdct
    @staticmethod
    def convertDocument2History(memorie):
        role,content = memorie.split(':')[0]
        return {"role":role,"content":content}      
    def process_memories(self,main_dct,message):
        memories = self._database.extractChunkFromDB(message=message)
        if memories  == {}:
            return (None,) * 4
        #dates = [memorie["date"] for memorie in memories]
        
        logging.error('awawa')
        logging.warn(memories)
        conversation = '\n'.join(memories["documents"])#[ x for conversation,date in zip(memories['documents'],dates)])
        
        summary = memories["metadatas"][0]["sumarization"]
        sentimental_conversation = memories["metadatas"][0]["sentimental_conversation"]
        # first date
        date = memories["metadatas"][0]["date"]
        
        return conversation,summary,sentimental_conversation,date
        # al ser un bloque solo el primero sera relevante
        
    def inject_llama_memories(self,main_dct:list[str],message:str,promptSystem) -> None:
        #self.get_memories = message
        # colocamos el query
        #memories = self.get_memories
        #memories = self._database.extractChunkFromDB(message=self._memories_message)
        conversation,summary,sentimental_conversation,date = self.process_memories(main_dct,message)
        
        if all((conversation,summary,sentimental_conversation,date)):
            return False 
        prp = promptSystem.format(
            character_ia=self._prompt_document.ia_prefix,
            user=self._prompt_document.user_prefix,
            location=self._prompt_document.scenario,
            conversation=conversation,# respuesta solo de ranni o de blake
            summary=summary,
            sentimental=sentimental_conversation,
            date=date
        )
        logging.error('REMEMBER INJECTED'.center(50,'-'))
        logging.error(prp)
        main_dct.append({'role':'system',"content":prp})
        # no hay memorias que procesar
    def inject_transformers_memories(self,past_dialogue,message):
        print(True)
        memories = self._database.extractChunkFromDB(message)
        logging.error('mem'.center(50,'='))
        logging.error(memories)
        
        if memories != {}:
            conversation = '\n'.join(memories["documents"])#[ x for conversation,date in zip(memories['documents'],dates)])
            summary = memories["metadatas"][0]["sumarization"]
            sentimental_conversation = memories["metadatas"][0]["sentimental_conversation"]
            # first date
            date = memories["metadatas"][0]["date"]
            past_dialogue.append(self._prompt_memories.format(
                character_ia=self._prompt_document.ia_prefix,
                user=self._prompt_document.user_prefix,
                location=self._prompt_document.scenario,
                conversation=conversation,
                summary=summary,
                sentimental=sentimental_conversation,
                date=date))
            logging.error('INJECT MEMORIES'.center(50,'-'))
            logging.error(past_dialogue)
    def prompt_gen_chat(self, message_history, message):
        past_dialogue = []
        for message_pairs in message_history:
            message1, message2 = message_pairs
            #print(message1,message2)
            past_dialogue.append(f"{message1['role']}: {message1['content']}")
            past_dialogue.append(f"{message2['role']}: {message2['content']}")
        #result = self._database._collection.query(
        #    query_texts=[message],# buscmos datos referentes al nuevo prompt e insertamos resultados
        #    n_results=self._database._chroma_config.top_predictions,
        #)
        #logging.info("QUERY".center(50,"="))
        #logging.warn(result)
        
        self.inject_transformers_memories(past_dialogue=past_dialogue,message=message)
        #if len(result["documents"]) != 0: # si hay un resultado
        #    past_dialogue.extend(result["documents"][0])# primer documento
        #    
        past_dialogue_formatted = "\n".join(past_dialogue)
        # inject_chat = lambda: alpaca_prompt if self.inject_chat_prompt else ""
        chat_formatter = TRANSFORMER_CHAT_FORMAT.get(self._chatSettings.chat_format,None)
        #logging.error(chat_formatter)
        assert chat_formatter # uf its none raises an error
        
        prompt = chat_formatter(
            user_alias=self.user_alias,
            character_name=self.character_name,
            personality=self._prompt_document.personality,
            text_example=self._prompt_document.text_example,
            conv_prompt=self.conv_prompt,
            past_dialogue_formatted=past_dialogue_formatted,
            message=message
        )
        #print(prompt)
        self.get_prompt = prompt
        return prompt
    def run(self,message):
            print(self.update_conversation(message=message))
            #self.reset_message_history()

class ChatPygmalionEmbebed:
    ''' embebed version of  pygmalion for low resources ''' 
    def __init__(self, message_history=[],inject_chat_prompt=True):
        super().__init__(message_history,inject_chat_prompt)

if __name__ == "__main__":#
    chat_instance = Chat(message_history=[])
    while True:
        pr = input(">>>")
        match pr:
            case 'exit':
                break
            case 'get_vdb':
                print(chat_instance.databasec.collection.peek())
                continue
            case 'get_timer_vs':
                print(len(chat_instance.message_history)%chat_instance._chatSettings.hook_storage)
                continue
            case 'get_history':
                print('CURRENT HISTORY'.center(50,'#'))
                print(chat_instance.message_history)
                continue
            case 'get_prompt':
                print('CURRENT PROMPT'.center(50,"#"))
                print(chat_instance.get_prompt)
                continue
            case 'help':
                print('\n'.join(["get_vdb","get_timer_vs","get_history","get_prompt"]))
                continue
            case '':
                continue
            case _:
                pass
            
        chat_instance.run(pr)
    # chat_instance.launch_gradio()