''' 

este script esta destinado a fungir como 
cli para el ususario

para crear documentos de modelos
'''

import click

from os.path import isfile,isdir,join
GLOBAL_HANDLER = None
### MODEL CREATOR TOOL ###

@click.command()
@click.argument('context')
@click.argument('ia_prefix')
@click.argument('user_prefix')
@click.option('--text_example',default='')
@click.argument('personality')
@click.argument('motions')
def createPromptDocument():
    pass
@click.command()

def model():
    pass

@click.command()
@click.argument('document')
def hyperDbRaw2Database(document:str):
    if not(isfile(document)):
        click.echo('Error the file does not exists',err=True,color=True,fg='red')
        return 0
    from ..hyperdb_handler import RawDocumentMerger
