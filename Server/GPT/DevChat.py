#import gradio as gr
#import torch
from models import ModelLoader,ChatBotSettings,Settings,Google_trans,Baidu_trans,PromptDocument,ChomaDBHandler,ChromaDb,ChromaDbClient,join,Metadata
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
        
        self._a = ''
        #self._database = None
        for message_pairs in message_history:
            message1, message2 = message_pairs
            self.display_messages.append([message1['text'], message2['text']])
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
        if self._chatSettings.use_summarysation:
            self.summarizator = lambda input_data: self.generator(input_data,task="summarization",min_lengt=self._chatSettings.min_sumarization_lengt,max_length=self._chatSettings.max_sumarization_lengt)
            
        
    def loadModel(self):
        match self._chatSettings.backend:
            case "transformers":
                self.transformer_backend()
            case "llamacpp":
                pass
            case "gpt4all":
                pass

    def load_settings(self):
        with ModelLoader("bot_settings.json",ChatBotSettings) as model:
            self._chatSettings = model
        with ModelLoader(configuration_name=self._chatSettings.full_prompt_document,ModelClass=PromptDocument,no_join_config_file_path=True) as ml:
            self._prompt_document = ml
            self.character_name = ml.ia_prefix
            self.user_alias = ml.user_prefix
            # NOTE: if a temp was defined it will replace the default configuration temperature
            self._chatSettings.temperature = ml.temp
        # INIT DATABASE
        self._database = ChomaDBHandler(self._prompt_document.ia_prefix)
        self._database.handler()
    
    def gpt_neo(self,prompt):
        return self.generator(prompt, do_sample=True, min_length=20, max_length=self._chatSettings.max_new_tokens)
         
    @property
    def get_prompt(self):
        return self._a
    @get_prompt.setter
    def get_prompt(self,arg):
        self._a = arg
    def evaluate(self, message, **kwargs):#  temperature=0.6, top_p=0.75, top_k=50, num_beams=5, max_new_tokens=256, repetition_penalty=1.4,
        prompt = self.prompt_gen_chat(self.message_history, message)
        self.get_prompt = prompt
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
            [ [metha.extend([x.get("methadata",False),y.get("methadata",False)]),doc.extend([f"{x['speaker']}: {x['text']}",f"{y['speaker']}: {y['text']}"])] for x,y in self.storage_hook]
            # sumarizamos
            print(doc)
            #self.summaryse_previus_conversation(doc)
            #self.sentimental_analysis(self.storage_hook)
            
            
            #[ sh.extend([print(x,y)]) for x,y in self.storage_hook]
            metha = [ x if x else Metadata(date=str(datetime.now())) for x in metha]
            # limpiamos datos para guardarlos
            #print(doc)#self.storage_hook[:self._chatSettings.hook_storage])
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
            lysis = self.conv_analysis([x["text"],y["text"]])
            conversation["methadata"]["sentimental_conversation"] = lysis[0]['label']
            conversation["methadata"]["sentimental_conversation"] =lysis[1]['label']
            # https://huggingface.co/docs/transformers/v4.36.1/en/quicktour#pipeline
            
    def update_conversation(self,message:str):
        ''' update the conversation add and format the messages ''' 
        response = self.evaluate(message)
        self.message_history.append(
            (
                {"speaker": self.user_alias, "text": message},
                {"speaker": self.character_name, "text": response},
            )
        )
        self.storage_hook.append(
            (
                {"speaker": self.user_alias, "text": message,"methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}},
                {"speaker": self.character_name, "text": response,"methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}},
            )
        )# usar el metodo copy demandaria mayor gasto de recursos
        #self.message_history.copy()
        # append the new message
        
        display = self.reset_message_history(message,response)
        return display
    def summaryse_previus_conversation(self,conversation:dict[str,str]):
        if self._chatSettings.use_summarysation:
            print(conversation)
            output = self.summarizator(conversation)
            for x,y in conversation:
                y['methadata']['sumarization'] = output
                x['methadata']['sumarization'] = output
            
        
        pass
    @property
    def databasec(self):
        return self._database
    def prompt_gen_chat(self, message_history, message):
        past_dialogue = []
        for message_pairs in message_history:
            message1, message2 = message_pairs
            past_dialogue.append(f"{message1['speaker']}: {message1['text']}")
            past_dialogue.append(f"{message2['speaker']}: {message2['text']}")
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
        return prompt
    def run(self,message):
            print(self.update_conversation(message=message))
            #self.reset_message_history()
    #def launch_gradio(self):
    #with gr.Blocks(theme="JohnSmith9982/small_and_pretty") as demo:
    #    chatbot = gr.Chatbot(elem_id="chatbot")
    #    with gr.Row():
    #        txt = gr.Textbox(show_label=False,
    #                         placeholder="Enter text and press enter")
    #    txt.submit(self.gradio_helper, txt, chatbot)
    #    txt.submit(lambda: "", None, txt)
    #
    #demo.launch(debug=True, share=True)

if __name__ == "__main__":#
    '''
    message_history = [
        (
            {
                "speaker": "Bob",
                "text": "Hey, Alice! How are you doing? What's the status on those reports?",
                "methadata": {"date":str(str(datetime.now())),"sentimental_conversation":'',"sumarization":''}
            },
            {
                "speaker": "Alice",
                "text": "Hey, Bob! I'm doing well. I'm almost done with the reports. I'll send them to you by the end of the day.",
                "methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''} 
            },
        ),
        (
            {
                "speaker": "Bob",
                "text": "That's great! Thanks, Alice. I'll be waiting for them. Btw, I have approved your leave for next week.",
                "methadata": {"date":str(datetime.now()),"sentimental_conversation":'',"sumarization":''}
            },
            {
                "speaker": "Alice",
                "text": "Oh, thanks, Bob! I really appreciate it. I will be sure to send you the reports before I leave. Anything else you need from me?",
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
