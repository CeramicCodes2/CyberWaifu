# import argparse, sys, os
import json, requests, random, hashlib, urllib.parse, base64
from time import sleep

from flask import Flask
# from flask import request
from markupsafe import escape
from translatepy.translators.google import GoogleTranslate
from models import ModelLoader,ChatBotSettings,Settings,Google_trans,Baidu_trans,PromptDocument
from transformers import pipeline,Conversation
class Config:
    config_files = {'settings': 'config/settings.json', 'api': 'config/api.json','bot_settings':'config/ChatBotSettings.json'}
    dicts = {'settings': {}, 'api': {}}

    def __init__(self):
        with open(self.config_files['settings'], 'r') as f:
            self.dicts['settings'] = json.loads(f.read())
        with open(self.config_files['api'], 'r') as f:
            self.dicts['api'] = json.loads(f.read())

    def read(self, file, key):
        return self.dicts[file][key]


class Utils:

    def url_text_encode(self, text):
        text = urllib.parse.quote(text, 'utf-8')
        reserved_replace = [['!', '%21'], ['#', '%23'], ['$', '%24'], ['&', '%26'], ['\'', '%27'], ['(', '%28'],
                            [')', '%29'], ['*', '%2A'], ['+', '%2B'], [',', '%2C'], ['/', '%2F'], [':', '%3A'],
                            [';', '%3B'], ['=', '%3D'], ['?', '%3F'], ['@', '%40'], ['[', '%5B'], [']', '%5D']]
        for i in reserved_replace:
            text = text.replace(i[0], i[1])
        return text

    def translate_baidu(self, query, target, source='en'):
        raw_text = query
        app_id = config.read('api', 'appid')
        key = config.read('api', 'key')
        salt = str(random.randint(1000000000, 9999999999))  # 文档中示例的salt为十位随机数
        sign = app_id + query + salt + key  # 签名中的query不进行百分号编码
        sign = hashlib.md5(sign.encode('utf-8')).hexdigest()  # 三步缺一不可

        api = 'https://fanyi-api.baidu.com/api/trans/vip/translate?q={}&from={}&to={}&appid={}&salt={}&sign={}'.format(
            self.url_text_encode(query), source, target, app_id, salt, sign)  # 请求URL中的query需要进行百分号编码

        i = 0
        code = 0
        max_retry = 10
        while code != 200:  # 没有成功就一直重试
            if i == 0:
                print(code)
            else:
                print('Retry {}'.format(str(i)))
                sleep(1)

            try:
                r = requests.get(api)
                code = r.status_code
            except Exception as e:
                code = -1
                print(e.args)
            print(code)

            if i > max_retry:
                print('Failed')
                return raw_text
            i = i + 1

        j = json.loads(r.text)
        # print(j)
        if 'error_code' in j:
            print("Translation error. code:{}, msg:{}".format(j['error_code'], j['error_msg']))
            return raw_text
        return j['trans_result'][0]['dst']

    def translate_google(self, query, target, source='eng'):
        trans = GoogleTranslate()
        while_ = True
        i = 0
        r = query
        while while_:
            try:
                r = trans.translate(text=query, destination_language=target, source_language=source)
                while_ = False
            except Exception as e:
                print(e.args)
                while_ = True
            if i > 0:
                sleep(3)
            i = i + 1
        return str(r)

    def trans(self, text, which):
        translator = config.read('settings', 'translator')
        if which == 'ipt':
            if translator == 'google':
                r = utils.translate_google(text, 'eng', config.read('settings', 'google-trans')['trans-ipt'])
                return r
            elif translator == 'baidu':
                r = utils.translate_baidu(text, 'en', config.read('settings', 'baidu-trans')['trans-ipt'])
                return r
            else:
                return text
        elif which == 'opt':
            if translator == 'google':
                target = config.read('settings', 'google-trans')['trans-opt']
                r = utils.translate_google(text, target)
                return r
            elif translator == 'baidu':
                target = config.read('settings', 'baidu-trans')['trans-opt']
                r = utils.translate_baidu(text, target)
                return r
            else:
                return text
        elif which == 'opt2':
            if translator == 'google':
                target = config.read('settings', 'google-trans')['trans-opt2']
                r = utils.translate_google(text, target)
                return r
            elif translator == 'baidu':
                target = config.read('settings', 'baidu-trans')['trans-opt2']
                r = utils.translate_baidu(text, target)
                return r
            else:
                return text
        else:
            return text

class ChatBot:
    '''
    available backend -> [
        gpt4all,
        transformers,
        oobabooga_API
    ]    
    '''
    def __init__(self):
        with ModelLoader("bot_settings.json",ChatBotSettings) as model:
            self._model = model
        self._iteration:int = 1
        # developent attrs
        #self.prompt_n = {'role':self._model.ia_prefix,"content":"My friend asked me"}
        self._conversation = []# first message
        self.sumarize = ""
        self._txt = ""
        self.prompt = ""
        self._sumarizePipeline = ""
        self.add_message = lambda role,content: {"role":role,"content":content}
        # use this function to add a message to conversation list
    @property
    def txt(self):
        return self._txt
    @txt.setter
    def txt(self,arg):
        self._txt = arg
    @txt.deleter
    def txt(self):
        self._txt = ""  
    @property
    def conversation(self):
        return self._conversation
    @conversation.setter
    def conversation(self,arg:dict[str,str]):
        if not(list(arg.keys()) == ["role","content"]):
            raise ValueError("Dictonary Error bad chat argumentsn !")
        self._conversation.append(arg)
    @property
    def iteration(self) -> int:
        return self._iteration
    @property
    def updateIteration(self):
        ''' update the iteration '''
        self._iteration += 1
    @staticmethod
    def pre_process_json(text):  # 预处理将被放入JSON的数据，使用urlsafe_base64编码
        text = base64.urlsafe_b64encode(text.encode('UTF-8')).decode('UTF-8')
        return text
    def loadTransformerOldSettings(self):
        kwargs_dct = dict()
        with ModelLoader(configuration_name="test.json",ModelClass=Settings) as ml:
            self._settings = ml
        if getattr(self._settings,self._settings.translator+'_trans',None) is not None:  # 翻译输入
            print(":: Translating input...")
            self.prompt = utils.trans(self.prompt, 'ipt')
            print(self.prompt)
        if self._settings.conversation is True:  # 对话模式
            self.prompt = "My friend asked me \"{}\", I answered: \"".format(self.prompt)
        if self._settings.useAccelerateModule:
                kwargs_dct["device_map"] = "auto"
        self.generator = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, **kwargs_dct)
    def newSettings(self):
        # carga modelo y procesa
        with ModelLoader(configuration_name="test.json",ModelClass=Settings) as ml:
            self._settings = ml
        with ModelLoader(configuration_name=self._model.full_prompt_document,ModelClass=PromptDocument) as ml:
            self._prompt_document = ml
            # carga del prompt
        if self._settings.useAccelerateModule:
            kwargs_dct["device_map"] = "auto"# use module accelerate
        if getattr(self._settings,self._settings.translator+'_trans',None) is not None:  # 翻译输入
            print(":: Translating input...")
            self.prompt = utils.trans(self.prompt, 'ipt')
            print(self.prompt)
        if self._settings.conversation:
            kwargs_dct["task"] = "conversational"
            self.conversation = self.add_message("system",self._prompt_document.context)
            # add the context
        if not(self._settings.conversation):
            kwargs_dct["task"] = 'text-generation'
             
        self.generator = pipeline(**kwargs_dct)
        # parse the configuration
        
    def useNewTransformerBackend(self): 
        self.newSettings()
        while True:
            self.updateIteration
            print(":: Generating...")
            min_length = self._settings.length * iteration
            max_length = self._settings.length * 2 * iteration
            self.conversation = {"role":self._prompt_document.ia_prefix,"content":self._prompt_document.context}
            print(self.conversation)
            result = self.generator(self.conversation, do_sample=True, min_length=min_length, max_length=max_length)
            #self.generator.tokenizer.default_chat_template set in the future the default template
            print(result)
            txt_opt = result[0]['generated_text']
            
            self.txt = txt_opt
            if self._settings.sentence: # true
                last_dot = self.txt.rfind('.')
                if last_dot == -1:
                    self.prompt = txt_opt
                    continue
                self.txt = self.txt[0:last_dot + 1]
                # continue the generation if in the oration hasn't the first dot
                
            if self._settings.no_lf:  # 忽略换行 | ignore line break's
                self.txt = self.txt.replace('\n', '')
            if self._settings.no_prompt:  # 丢弃输出中的prompt
                # "My friend asked me \"Hola como estas hoy ?\", I answered: \"bien y tu ?"
                print(self.prompt)
                print(self.txt)
                self.txt = self.txt.replace(self.prompt, '')
                # \"bien y tu ?"
                quote = [self.txt.find('\"')]
                if self._settings.answer_only:  # 仅保留回答内容 | solo conserva respuestas
                    if quote[0] == -1:
                        self.prompt = txt_opt
                        continue
                    self.txt = self.txt[0:quote[0]]
                else:
                    if quote[0] == -1:
                        self.txt = self.txt
                    else:
                        self.txt = '\'' + self.txt
            # tr = config.read('settings', config.read('settings', 'translator') + '-trans')['trans-opt']

            tr:Google_trans | Baidu_trans = getattr(self._settings,self._settings.translator+'_trans',None)
            #self.text = "h1"
            if tr is not None:  # 翻译输出
                print(":: Translating output to {}...".format(tr.trans_opt))
                txt_trans = utils.trans(self.txt, 'opt')

                #tr_2 = config.read('settings', config.read('settings', 'translator') + '-trans')['trans-opt2']
                tr_2:Google_trans | Baidu_trans = getattr(self._settings,self._settings.translator+'_trans',None)
                # check the other translation option
                if tr_2 is not None:  # 翻译输出2
                    print(":: Translating output to {}...".format(tr_2.trans_opt2))
                    txt_trans_2 = utils.trans(self.txt, 'opt2')
                    # 返回 JSON
                    return json.dumps({
                        "raw": self.pre_process_json(self.txt),
                        tr: self.pre_process_json(txt_trans),
                        tr_2: self.pre_process_json(txt_trans_2)
                    })

                # 返回 JSON
                return json.dumps({
                    "raw": self.pre_process_json(self.txt),
                    tr: self.pre_process_json(txt_trans)

                })
            return json.dumps({
            "raw": pre_process_json(self.txt)
        })
    def useTransformerBackend(self):
        """ 
        this method will implement the old logic used in the original repository
        and add the new features like Sumarization and basic memory
        """
        #self.loadTransformerOldSettings()
        
        while True:
            self.updateIteration
            print(":: Generating...")
            min_length = self._settings.length * iteration
            max_length = self._settings.length * 2 * iteration
            result = self.generator(self.prompt, do_sample=True, min_length=min_length, max_length=max_length)
            print(self.generator.tokenizer.default_chat_template)
            txt_opt = result[0]['generated_text']
            self.txt = txt_opt
            if self._settings.sentence: # true
                last_dot = self.txt.rfind('.')
                if last_dot == -1:
                    self.prompt = txt_opt
                    continue
                self.txt = self.txt[0:last_dot + 1]
                # continue the generation if in the oration hasn't the first dot
                
            if self._settings.no_lf:  # 忽略换行 | ignore line break's
                self.txt = self.txt.replace('\n', '')

            if self._settings.no_prompt:  # 丢弃输出中的prompt
                # "My friend asked me \"Hola como estas hoy ?\", I answered: \"bien y tu ?"
                print(self.prompt)
                print(self.txt)
                self.txt = self.txt.replace(self.prompt, '')
                # \"bien y tu ?"
                quote = [self.txt.find('\"')]
                if self._settings.answer_only:  # 仅保留回答内容 | solo conserva respuestas
                    if quote[0] == -1:
                        self.prompt = txt_opt
                        continue
                    self.txt = self.txt[0:quote[0]]
                else:
                    if quote[0] == -1:
                        self.txt = self.txt
                    else:
                        self.txt = '\'' + self.txt
            # tr = config.read('settings', config.read('settings', 'translator') + '-trans')['trans-opt']
            tr:Google_trans | Baidu_trans = getattr(self._settings,self._settings.translator+'_trans',None)
            if tr is not None:  # 翻译输出
                print(":: Translating output to {}...".format(tr.trans_opt))
                txt_trans = utils.trans(self.txt, 'opt')

                #tr_2 = config.read('settings', config.read('settings', 'translator') + '-trans')['trans-opt2']
                tr_2:Google_trans | Baidu_trans = getattr(self._settings,self._settings.translator+'_trans',None)
                # check the other translation option
                if tr_2 is not None:  # 翻译输出2
                    print(":: Translating output to {}...".format(tr_2.trans_opt2))
                    txt_trans_2 = utils.trans(self.txt, 'opt2')
                    # 返回 JSON
                    return json.dumps({
                        "raw": self.pre_process_json(self.txt),
                        tr: self.pre_process_json(txt_trans),
                        tr_2: self.pre_process_json(txt_trans_2)
                    })

                # 返回 JSON
                return json.dumps({
                    "raw": self.pre_process_json(self.txt),
                    tr: self.pre_process_json(txt_trans)

                })
            return json.dumps({
            "raw": pre_process_json(self.txt)
        })
    def gpt4AllBackend(self):
        raise NameError('NOT IMPLEMENTED !')
    def run(self):
        match self._model.backend:
            case 'gpt4all':
                self.gpt4AllBackend()
            case 'transformers':
                # creates the pipeline
                self.loadTransformerOldSettings()
                #self.generator = pipeline('text-generation', model=self._model.model_path)
                self.useTransformerBackend()
            case 'transformers_nbackend':
                #self.newSettings()
                return self.useNewTransformerBackend()
                
def process(prompt):
    # 前处理
    if config.read('settings', config.read('settings', 'translator') + '-trans')['trans-ipt'] is not None:  # 翻译输入
        print(":: Translating input...")
        prompt = utils.trans(prompt, 'ipt')
        print(prompt)

    if config.read('settings', 'conversation') is True:  # 对话模式
        prompt = "My friend asked me \"{}\", I answered: \"".format(prompt)
    
    #ChatBot().loadTransformerOldSettings()
    # 处理
    i = 1
    while True:
        i = i + 1
        print(":: Generating...")
        length = config.read('settings', 'length')  # 文本长度
        min_length = length * i
        max_length = length * 2 * i
        print(generator.tokenizer.default_chat_template)
        result = generator(prompt, do_sample=True, min_length=min_length, max_length=max_length)  # 生成
        txt_opt = result[0]['generated_text']
        txt = txt_opt

        # 后处理
        if config.read('settings', 'sentence') is True:  # 截取整句
            last_dot = txt.rfind('.')
            if last_dot == -1:
                prompt = txt_opt
                continue
            txt = txt[0:last_dot + 1]

        if config.read('settings', 'no-lf') is True:  # 忽略换行
            txt = txt.replace('\n', '')

        if config.read('settings', 'no-prompt') is True:  # 丢弃输出中的prompt
            print(prompt)
            print(txt)
            txt = txt.replace(prompt, '')
            quote = [txt.find('\"')]
            if config.read('settings', 'answer-only') is True:  # 仅保留回答内容
                if quote[0] == -1:
                    prompt = txt_opt
                    continue
                txt = txt[0:quote[0]]
            else:
                if quote[0] == -1:
                    txt = txt
                else:
                    txt = '\'' + txt

        def pre_process_json(text):  # 预处理将被放入JSON的数据，使用urlsafe_base64编码
            text = base64.urlsafe_b64encode(text.encode('UTF-8')).decode('UTF-8')
            return text
        
        tr = config.read('settings', config.read('settings', 'translator') + '-trans')['trans-opt']
        if tr is not None:  # 翻译输出
            print(":: Translating output to {}...".format(tr))
            txt_trans = utils.trans(txt, 'opt')

            tr_2 = config.read('settings', config.read('settings', 'translator') + '-trans')['trans-opt2']
            if tr_2 is not None:  # 翻译输出2
                print(":: Translating output to {}...".format(tr_2))
                txt_trans_2 = utils.trans(txt, 'opt2')
                # 返回 JSON
                return json.dumps({
                    "raw": pre_process_json(txt),
                    tr: pre_process_json(txt_trans),
                    tr_2: pre_process_json(txt_trans_2)
                })

            # 返回 JSON
            return json.dumps({
                "raw": pre_process_json(txt),
                tr: pre_process_json(txt_trans)

            })

        # if sys.platform == 'win32':
        #     os.system('cls')
        # else:
        #     os.system('clear')

        # 返回 JSON
        return json.dumps({
            "raw": pre_process_json(txt)
        })
        # return txt


def main():
    global config
    global utils
    global generator
    
    config = Config()
    utils = Utils()
    
    
    #ChatBot().loadTransformerOldSettings()
    # print(":: Testing Google translate...")
    # print(utils.translate_google('hello', 'zh', 'eng'))
    # print(":: Testing Baidu translate...")
    # print(utils.translate_baidu('hello', 'zh', 'en'))

    print(":: Importing module...")
    #from transformers import pipeline  # 导入模块

    print(":: Loading model...")
    generator = pipeline('conversational', model=r'model/gpt-neo-125m') # 'text-generation'  # 加载模型
    pr = ChatBot()
    pr.run()
    app = Flask(__name__)

    @app.route("/<text>")
    def receive_request(text):
        print(text)
        return escape(pr(text))

    app.run(host='127.0.0.1', port=7210)


if __name__ == '__main__':
    main()
