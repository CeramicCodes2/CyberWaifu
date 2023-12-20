import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from models import ModelLoader,ChatBotSettings,Settings,Google_trans,Baidu_trans,PromptDocument


class Chat:
    def __init__(self, modelName:str, tokenizer, conv_prompt, user_alias='User', character_name='Chatbot', message_history=[], chat_buffer_size=10):
        self.modelName = modelName
        self.load_settings()
        self.conv_prompt = self._prompt_document.context# loads the context of the situation
        
        self.user_alias = self._prompt_document.user_prefix
        self.character_name = self._prompt_document.ia_prefix
        self.chat_buffer_size = self._chatSettings.chat_buffer_size
        self.message_history = message_history
        self.display_messages = []# mensajes a mostrar
        for message_pairs in message_history:
            message1, message2 = message_pairs
            self.display_messages.append([message1['text'], message2['text']])
        self.loadModel()
    def loadModel(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self._chatSettings.model_path + self.modelName, device_map="auto", load_in_8bit=load_in_8bit)
        self.tokenizer = LlamaTokenizer.from_pretrained(self._chatSettings.model_path + self.modelName)
    def load_settings(self):
        with ModelLoader("bot_settings.json",ChatBotSettings) as model:
            self._chatSettings = model
        with ModelLoader(configuration_name="test.json",ModelClass=Settings) as ml:
            self._settings = ml
        with ModelLoader(configuration_name=self._model.full_prompt_document,ModelClass=PromptDocument) as ml:
            self._prompt_document = ml
            self.character_name = ml.ia_prefix
            self.user_alias = ml.user_prefix
            # NOTE: if a temp was defined it will replace the default configuration temperature
            self._chatSettings.temperature = ml.temp
    def evaluate(self, message, **kwargs):#  temperature=0.6, top_p=0.75, top_k=50, num_beams=5, max_new_tokens=256, repetition_penalty=1.4,
        prompt = self.prompt_gen_chat(self.message_history, message)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        generation_config = GenerationConfig(
            temperature=self._chatSettings.temperature,
            top_p=self._chatSettings.top_p,
            top_k=self._chatSettings.top_k,
            num_beams = self._chatSettings.num_beams,
            early_stopping=self._chatSettings.early_stopping,
            repetition_penalty=self._chatSettings.repetition_penalty
            #temperature=temperature,
            #top_p=top_p,
            #top_k=top_k,
            #num_beams=num_beams,
            #early_stopping=True,
            #repetition_penalty=repetition_penalty,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self._chatSettings.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        split_str = """### Response:\n{self.character_name}:"""
        output = output.split(split_str)[1].strip()
        return output
    def reset_message_history(self,save_overflow_conversation=True):
        ''' 
        argument save_overflow_conversation will save the conversation using a vector storage database
        {
         "sumarization":"",
         "messages":{
             
         }   
        }
        '''
        if(len(self.message_history) > self.chat_buffer_size):
            self.message_history = self.message_history[-self.chat_buffer_size:]
            # toma del ulimo elemento en adelante
        self.display_messages.append([message, response])
        return self.display_messages
    def update_conversation(self,message:str,response:str):
        ''' update the conversation add and format the messages ''' 
        response = self.evaluate(message)
        self.message_history.append(
            (
                {"speaker": self.user_alias, "text": message},
                {"speaker": self.character_name, "text": response},
            )
        )
        # append the new message
        self.reset_message_history()
    def summaryse_previus_conversation(self):
        match self._chatSettings.vectorStorageBackend:
            case "Chomadb":
                
                ...
            case _:
                raise NameError("Error vector storage db not implemented !") 
    def save_conversation(self):
        """ this method will save the data using a vectorstorage database"""
        pass
    def search_conversation(self,user_message:str):
        pass
    def gradio_helper(self, message):
        # make response
        response = self.evaluate(message)
        # update message history
        self.message_history.append(
            (
                {"speaker": self.user_alias, "text": message},
                {"speaker": self.character_name, "text": response},
            )
        )
        if len(self.message_history) > self.chat_buffer_size:
            self.message_history = self.message_history[-self.chat_buffer_size:]
        # update display messages
        self.display_messages.append([message, response])
        return self.display_messages

    def prompt_gen_chat(self, message_history, message):
        past_dialogue = []
        for message_pairs in message_history:
            message1, message2 = message_pairs
            past_dialogue.append(f"{message1['speaker']}: {message1['text']}")
            past_dialogue.append(f"{message2['speaker']}: {message2['text']}")
        past_dialogue_formatted = "\n".join(past_dialogue)

        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{self.conv_prompt}

This is the conversation between {self.user_alias} and {self.character_name} till now:
{past_dialogue_formatted}

Continuing from the previous conversation, write what {self.character_name} says to {self.user_alias}:
### Input:
{self.user_alias}: {message}
### Response:
{self.character_name}:"""

        return prompt
    def run(self,message):
            self.update_conversation(message=message)
            self.reset_message_history()
    def launch_gradio(self):
        with gr.Blocks(theme="JohnSmith9982/small_and_pretty") as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            with gr.Row():
                txt = gr.Textbox(show_label=False,
                                 placeholder="Enter text and press enter")
            txt.submit(self.gradio_helper, txt, chatbot)
            txt.submit(lambda: "", None, txt)

        demo.launch(debug=True, share=True)


if __name__ == "__main__":
    model_path = "Xilabs/calypso-3b-alpha-v2"
    load_in_8bit = False
    model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", load_in_8bit=load_in_8bit)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    conv_prompt = "Two people are texting each other on a messaging platform."
    message_history = [
        (
            {
                "speaker": "Bob",
                "text": "Hey, Alice! How are you doing? What's the status on those reports?",
            },
            {
                "speaker": "Alice",
                "text": "Hey, Bob! I'm doing well. I'm almost done with the reports. I'll send them to you by the end of the day.",
            },
        ),
        (
            {
                "speaker": "Bob",
                "text": "That's great! Thanks, Alice. I'll be waiting for them. Btw, I have approved your leave for next week.",
            },
            {
                "speaker": "Alice",
                "text": "Oh, thanks, Bob! I really appreciate it. I will be sure to send you the reports before I leave. Anything else you need from me?",
            },
        )
    ]

    chat_instance = Chat(model, tokenizer, conv_prompt, user_alias='Bob',
                             character_name='Alice', message_history=message_history)
    chat_instance.launch_gradio()
