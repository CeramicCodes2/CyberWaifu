from subprocess import Popen,PIPE
#from threading import Thread
prp = Popen(['flask','--app','\Server\GPT\api','--debug','run'])
prp.wait()
