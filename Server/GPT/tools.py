# dev tools
from models import Tools,join,listdir,isfile,ModelDescription,EventPrompt,ModelLoader,PromptDocument,levelPrompt,GenericPrompt
class ToolLoader(ModelLoader):
    def __init__(self,configuration_name):
        super().__init__(configuration_name=configuration_name,ModelClass=EventPrompt,no_join_config_file_path=True)
        self.listExistingTools()
        self.NO_TOOLS_FLAG = False
    def listExistingTools(self):
        self.tools = [tool for tool in listdir('tools/')]
        if self.tools == []:
            self.NO_TOOLS_FLAG = True
            # usar para comprobar que no hay herramientas
            
class EPM:
    def __init__(self,prompt_EventToolSelector:str,prompt_document:PromptDocument,text_generator):#,tool_path:str,text_generator:str):
        self.evts = prompt_EventToolSelector# prompt 
        self.prompt_document = prompt_document
        self.text_generator = text_generator
        self.current_level:levelPrompt = {}
        self.tool_pool:list[EventPrompt] = []
        self.prpEventDict:dict[str,str] = {}
        self.processedTools:str = None
        self.tool_names = []
        self.search_format = lambda prompt: "user_prefix" if ("user_prefix" in prompt) else "ia_prefix" if ("ia_prefix" in prompt) else ''
        self.loadTools()
    def searchPrefix(self,prompt):
        prefixes:dict[str,str] = {}
        if ("user_prefix" in prompt):
            prefixes["user_prefix"] = self.prompt_document.user_prefix
        if ("ia_prefix" in prompt):
            prefixes["ia_prefix"] = self.prompt_document.ia_prefix
        return prefixes
    def loadTool(self,toolName:str):
        with ModelLoader(configuration_name=toolName,ModelClass=EventPrompt,no_join_config_file_path=True) as ml:
            #print(ml)
            self.tool_pool.append(ml)
        #try:
        #    print(toolName)
        #except Exception as e:
        #    print(f'THE TOOL {toolName} could not be loadded \n Error status: \n \t {e}')

            
    def loadTools(self):
        # se usa para cargar las herramientas
        self.tools = [join('tools',tool) for tool in listdir('tools/') if tool in self.prompt_document.enabled_events]
        # listamos las herramientas permitidas
        # ahora cargamos esas herramientas
        if self.tools == []:
            self.tool_pool = []
            return 
        [ self.loadTool(tool) for tool in self.tools]
        
    def processTools(self,current_level) -> list[str]:
        # dependiendo del nivel de intimidad se mostrara o no alguna herramienta dada
        for tool in self.tool_pool:
            if not(all([True if action in current_level['actions'] else False for action in tool.actionsAssociated])):
                # si todos no estan no se usara el evento
                continue
            tool.models = [ ModelDescription(**tl) for tl in tool.models] 
            yield self.createPromptForTool(tool=tool)
            # else
            
    def modelsPrompt(self,tool:EventPrompt) -> iter:
        # creara un prompt de los modelos
        for model in tool.models:
            # convertimos 
            model_prompt = []
            model_prompt.append("\t\t action: ")# + ':'
            if rsp:=self.searchPrefix(model.description):
                model.description = model.description.format(**rsp)
            if rsp:=self.searchPrefix(model.promptModel):
                model.promptModel = model.promptModel.format(**rsp)
            model_prompt.append(model.model_name)
            model_prompt.append(model.promptModel)
            yield ' '.join(model_prompt)
    def createPromptForTool(self,tool:EventPrompt):
        prompt = []
        self.tool_names.append(tool.name)
        prompt.append('tool: ' + tool.name)
        if rsp:=self.searchPrefix(tool.description):
            tool.description = tool.description.format(**rsp)
        prompt.append('\t description:' + tool.description)
        prompt.append('the following tool provide the follow actions expressions and motions: \n')
        # convertimos a model description
        prompt.append('\t'.join([x for x in self.modelsPrompt(tool)]))
        if rsp:=self.searchPrefix(tool.promptEvent):
            # TODO: se usara para canviar el world scenario
            
            tool.promptEvent = tool.promptEvent.format(**rsp)
            prpEventDict[tool.name] = tool.promptEvent
        return '\n '.join(prompt)
        ...
    def createPrompt(self,file_path:str,current_level:dict[str,str]):
        # prp = self.evts.format(ia_prefix=self.prompt_document.ia_prefix,user_prefix=self.prompt_document.user_prefix,tools=)
        # crea un prompt guardable como archivo usa para ello la clase generic prompt
        self.loadTools()
        with open(file_path,'w') as dmp:
            dmp.write(str(GenericPrompt(prompt='\n'.join([x for x in self.processTools(current_level=current_level)]),model_configs=self.prpEventDict)))
    def InferenceWithTextGenerator(self,messages,current_level,use_llama:bool=False):
        # funcion para generacion de inferencia
        # tipo stuff sumarization
        if not(self.processedTools):
            self.processedTools = '\n '.join([x for x in self.processTools(current_level=current_level) ])
        response = self.text_generator(self.evts.format(user_prefix=self.prompt_document.user_prefix,ia_prefix=self.prompt_document.ia_prefix,
                                                        messages=messages,
                                                        tools=self.processedTools),use_llama=use_llama)
        print('PRUEBA RESPUESTA'.center(50,'='))
        print(response)
        if use_llama:
            # se selecciono ninguna herramienta o evento
            response = response["choices"][0]["text"]
        else:
            response = response[0]["generated_text"]
        
        if not("void" in response):
            # si selecciono una herramienta
            selection = [selected for selected in self.tool_names if selected in response]
            # deberia de ser solo una 
            print('INFO'.center(50,"="))
            print(self.evts.format(user_prefix=self.prompt_document.user_prefix,ia_prefix=self.prompt_document.ia_prefix,
                                                        messages=messages,
                                                        tools=self.processedTools))
            print(selection)
            if selection != []:
                return selection[-1]
            
            
    def roleInference(self,main_dct:list[dict[str,str]]):
        '''
        con el historial de chat se le pedira a la ia que genere una eleccion en base al chat
        tipo
        {user} dijo esto
        que accion quieres tomar ?
        >- ir de cita
        >- besar
        >- responder 
        
        si se quiere hacer esto se pasara el generador despues de los mensajes
        
        se insertara un prompt especial en el indice 4 como system
        '''
        
        ...
        #main_dct.insert(4)
        
        
        
        
        
if __name__ == '__main__':
    from models import PromptDocument
    with ModelLoader(configuration_name=
                './prompt_paths/ranni.json',ModelClass=PromptDocument,no_join_config_file_path=True) as ml:
        pdoc = ml
    with ModelLoader(configuration_name='./prompt_paths/EventToolPrompt.json',ModelClass=GenericPrompt,no_join_config_file_path=True) as ml:
        pet = ml
    evtool = EPM(
        prompt_EventToolSelector=pet,
        prompt_document=pdoc,
        text_generator=lambda text: {'choice':['hello world']},
        current_level={}
    )
    #print(evtool.testTool)
    ...