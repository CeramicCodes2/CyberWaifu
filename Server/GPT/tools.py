# dev tools
from models import Tools,join,listdir,isfile
class ToolLoader:
    def __init__(self):
        self.listExistingTools()
        self.NO_TOOLS_FLAG = False
    def listExistingTools(self):
        self.tools = [tool for tool in listdir('tools/')]
        if self.tools == []:
            self.NO_TOOLS_FLAG = True
            # usar para comprobar que no hay herramientas