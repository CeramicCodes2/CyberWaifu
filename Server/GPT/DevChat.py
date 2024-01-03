#import gradio as gr
#import torch
from models import ModelLoader,ChatBotSettings,Settings,Google_trans,Baidu_trans,PromptDocument,ChomaDBHandler,ChromaDb,ChromaDbClient,join,Metadata,GenericPrompt
from datetime import datetime

class Chat:
    def __init__(self, modelName:str, message_history=[]):
        self.modelName = modelName
        self.load_settings()
        self.conv_prompt = self._prompt_document.context# .format(user=self._prompt_document.user_prefix)# loads the context of the situation
        
        self.user_alias = self._prompt_document.user_prefix
        self.character_name = self._prompt_document.ia_prefix
        self.chat_buffer_size = self._chatSettings.chat_buffer_size
        self.message_history = message_history
        self.display_messages = []# mensajes a mostrar
        self.storage_hook = self.message_history.copy()
        self.pairRegister2Block = lambda x,y: [f"{x['role']}: {x['content']}",f"{y['role']}: {y['content']}"]
        self._prompt = ''
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
        from transformers import LlamaForCausalLM, LlamaTokenizer,pipeline,GenerationConfig
        
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
        self.generator = pipeline('text-generation',do_sample=True,model=self._chatSettings.model_path,device_map="auto",max_length=self._chatSettings.max_new_tokens)
        self.conv_analysis = lambda input_data: self.generator(input_data,task="sentiment-analysis")
        #if self._chatSettings.use_summarysation:
        #    s#elf.summarizator = lambda input_data: self.generator(input_data,task="summarization",min_length=5, max_length=20)#max_length=self._chatSettings.max_sumarization_lengt)       
    def llamaCpp_backend(self):
        from llama_cpp import Llama,llama_tokenize
        self.generator = Llama(model_path=self._chatSettings.model_path,
                               chat_format="llama-2",
                               max_tokens=self._chatSettings.chat_buffer_size,
                               n_gpu_layers=-1,
                               main_gpu=0,
                               n_ctx=1024
                               )
    def loadModel(self):
        match self._chatSettings.backend:
            case "transformers":
                self.transformer_backend()
                self.evaluate = self.transformers_evaluate
            case "llamacpp":
                self.llamaCpp_backend()
                self.evaluate = self.llama_evaluate
            case "gpt4all":
                pass
            case "debug":
                self.generator =  lambda text: [{"generated_text":text}]

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
            self._prompt_sentymental = ml.prompt
        with ModelLoader(configuration_name=self._chatSettings.full_summarization_document,ModelClass=GenericPrompt,no_join_config_file_path=True) as ml:
            self._prompt_summarizator = ml.prompt
            
        # INIT DATABASE
        self._database = ChomaDBHandler(self._prompt_document.ia_prefix)
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
        output = self.generator(prompt)[0]["generated_text"]
        # output = self.gpt_neo(prompt=prompt)[0]["generated_text"]
        # generator = pipeline('conversational', model=r'model/gpt-neo-125m')
        split_str = f"""### Response:\n{self.character_name}:"""
        output = output.split(split_str)[1].strip()
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
            print("CALLING SUMMARIZATOR")
            self.summarizator()
            #self.sentimental_analysis(self.storage_hook)
            [ [metha.extend([x.get("methadata",False),y.get("methadata",False)]),doc.extend([f"{x['role']}: {x['content']}",f"{y['role']}: {y['content']}"])] for x,y in self.storage_hook]
            # sumarizamos
            print(doc)
            
            
            #[ sh.extend([print(x,y)]) for x,y in self.storage_hook]
            metha = [ x if x else Metadata(date=str(datetime.now())) for x in metha]
            # limpiamos datos para guardarlos
            #print(doc)#self.storage_hook[:self._chatSettings.hook_storage])
            print("SENTYMENTAL CALL".center(50,"#"))
            self.sentymental()
               
            self._database.createDocument(doc,metha=metha)#[:self._chatSettings.hook_storage]))
            # al activarse se guardan los primeros elementos
            
            print("HOOK".center(50,"#") + '\n',self.storage_hook)
            #print("HOOK".center(100,"#") + "\n",self.storage_hook)
        #self.storage_hook.append([message,response])
        self.display_messages.append([message, response])
        return self.display_messages
        #return self.display_messages
    def sentimental_analysis(self,conversation):
        for x,y in conversation:
            print(self.pairRegister2Block(x,y))
            lysis = self.conv_analysis(self.pairRegister2Block(x,y))
            conversation["methadata"]["sentimental_conversation"] = lysis[0]
            conversation["methadata"]["sentimental_conversation"] = lysis[1]
            # https://huggingface.co/docs/transformers/v4.36.1/en/quicktour#pipeline
            
    def update_conversation(self,message:str):
        ''' update the conversation add and format the messages ''' 
        response = self.evaluate(message)
        self.message_history.append(
            (
                {"role": self.user_alias, "content": message},
                {"role": self.character_name, "content": response},
            )
        )
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
    def summarizator(self):
        ''' stuff summarization requieres an llm https://python.langchain.com/docs/use_cases/summarization'''
        
        # guardar con todo el historial o solo los bloques especificos  ?
        # por ahora solo los  bloques especificados
        gblock = []
        for x,y in self.storage_hook:
            gblock.extend(self.pairRegister2Block(x,y))
        gblock = ' \n'.join(gblock)
        pre_summary = self._prompt_summarizator.format(messages=gblock,ia_prefix=self._prompt_document.ia_prefix,user_prefix=self._prompt_document.user_prefix)
        summary = self.generator(pre_summary)
        #print(summary)
        summary = summary[0]["generated_text"]
        #gblock = ' \n'.join([ self.pairRegister2Block(x,y) for x,y in block]) 
        # TODO GUARDAR EN EL self.storage_hook
        for idx,item in enumerate(self.storage_hook):
            x,y = item
            x['methadata']['sumarization'] = summary
            y['methadata']['sumarization'] = summary
            item = (x,y)
            print("summary",item)
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
            gen = self.generator(pre_prompt)[0]["generated_text"]
            x['methadata'][f'sentimental_conversation'] = gen
            y['methadata'][f'sentimental_conversation'] = gen
        #gblock = ' \n'.join([ self.pairRegister2Block(x,y) for x,y in block]) 
        # TODO GUARDAR EN EL self.storage_hook        
    @property
    def databasec(self):
        return self._database
    def llama_propt_gen_chat(self,message_history,message):
        main_dct = []
        prp = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n ### Instruction:\n{self.conv_prompt}\n{self._prompt_document.personality}\nThe following texts are an example of something you would say: \n{self._prompt_document.text_example}"""
        sysHist = f'''\n This is the conversation between {self.user_alias} and {self.character_name} till now: \n'''        
        usrInput = f"""\n Continuing from the previous conversation, write what {self.character_name} says to {self.user_alias}:\n### Input:\n{self.user_alias}: {message}\n### Response:\n{self.character_name}:"""
        main_dct.append({"role":"system","content":prp})
        main_dct.append({"role":"system","content":sysHist})
        main_dct.extend(self.pair2tuple(self.message_history))
        main_dct.append({"role":"system","content":usrInput})
        self.get_prompt = main_dct
        print(main_dct)
        return self.generator.create_chat_completion(
            messages=main_dct
        )
    def pair2tuple(self,messages:tuple[dict[str,str]]) -> list[dict[str,str]]:
        lstdct:tuple[dict[str,str]] = []
        for pairs in messages:
            x,y  = pairs
            lstdct.append(x)
            lstdct.append(y)
        return lstdct
            
            
    def prompt_gen_chat(self, message_history, message):
        past_dialogue = []
        for message_pairs in message_history:
            message1, message2 = message_pairs
            print(message1,message2)
            past_dialogue.append(f"{message1['role']}: {message1['content']}")
            past_dialogue.append(f"{message2['role']}: {message2['content']}")
        result = self._database._collection.query(
            query_texts=[message],# buscmos datos referentes al nuevo prompt e insertamos resultados
            n_results=self._database._chroma_config.top_predictions,
        )
        if len(result["documents"]) != 0: # si hay un resultado
            past_dialogue.extend(result["documents"][0])# primer documento
            
        past_dialogue_formatted = "\n".join(past_dialogue)
        

        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.


### Instruction:
{self.conv_prompt}

{self._prompt_document.personality}
The following texts are an example of something you would say
{self._prompt_document.text_example}

This is the conversation between {self.user_alias} and {self.character_name} till now:
{past_dialogue_formatted}

Continuing from the previous conversation, write what {self.character_name} says to {self.user_alias}:
### Input:
{self.user_alias}: {message}
### Response:
{self.character_name}:"""
        #print(prompt)
        self.get_prompt = prompt
        return prompt
    def run(self,message):
            print(self.update_conversation(message=message))
            #self.reset_message_history()


if __name__ == "__main__":#
    '''
    message_history = [
        (
            {
                "role": "Bob",
                "content": "Hey, Alice! How are you doing? What's the status on those reports?",
                "methadata": {"date":str(str(datetime.now())),"sentimental_conversation":'',"sumarization":''}
            },
            {
                "role": "Alice",
                "content": "Hey, Bob! I'm doing well. I'm almost done with the reports. I'll send them to you by the end of the day.",
                "methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''} 
            },
        ),
        (
            {
                "role": "Bob",
                "content": "That's great! Thanks, Alice. I'll be waiting for them. Btw, I have approved your leave for next week.",
                "methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}
            },
            {
                "role": "Alice",
                "content": "Oh, thanks, Bob! I really appreciate it. I will be sure to send you the reports before I leave. Anything else you need from me?",
                "methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}
            },
        )
    ]
    '''
    chat_instance = Chat(modelName="calypso-3b-alpha-v2",message_history=[])
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
            case '':
                continue
            case _:
                pass
            
        chat_instance.run(pr)
    # chat_instance.launch_gradio()
