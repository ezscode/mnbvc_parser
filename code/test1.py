

import os 
import time 
from parser_base import Parser


class ParserWHPJax(Parser):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__() 

        for n in args:
            print('-- arg : ', n) 

        for k, v in kwargs.items():
            print('-- ', k, v) 

        model_name = kwargs.get('model_name', '')
        print('-- model_name : ', model_name) 

        model_name = "openai/whisper-large-v3"
        # self.pipeline = FlaxWhisperPipline(model_name, dtype=jnp.bfloat16, batch_size=16)  
     
 

    def parse_file(self, file_path):
         print('-- ', file_path) 


parser = ParserWHPJax(a='1', b=2)
parser.parse_file('file_path1')


from parse_audio import ParseAudioAdmin