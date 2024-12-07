#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parser_whisperjax.py
@Time    :   2024-12-07 14:21:39
@Author  :   Ez
@Version :   1.0
@Desc    :   Whisper-Jax 处理

'''

import os 
import time 
from parser_base import Parser


import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline

class ParserWHPJax(Parser):

    def __init__(self, *args, **kwargs): 
        super().__init__() 

        model_name = kwargs.get('model_name', '')
        batch_size = kwargs.get('batch_size', 16)
        dtype = kwargs.get('dtype', 0) 
        if dtype == 0:dtype=jnp.bfloat16  
        model_name = model_name.strip()
        if len(model_name) == 0:model_name = "openai/whisper-large-v3"
 
        self.pipeline = FlaxWhisperPipline(model_name, dtype=dtype, batch_size=batch_size)  
       

    def parse_file(self, file_path):
           
        ret_dict = self.pipeline(file_path, task="transcribe", return_timestamps=True) 
        return ret_dict  
    

