''' 

este script esta destinado a fungir como 
cli para el ususario
'''

import click

from os.path import isfile,isdir,join
GLOBAL_HANDLER = None
@click.command()
@click.argument('document')
def hyperDbRaw2Database(document:str):
    if not(isfile(document)):
        click.echo('Error the file does not exists',err=True,color=True,fg='red')
        return 0
    from ..hyperdb_handler import RawDocumentMerger
