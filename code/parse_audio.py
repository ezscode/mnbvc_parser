#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parse_audio.py
@Time    :   2024-12-07 14:14:17
@Author  :   Ez
@Version :   1.0
@Desc    :   音频处理
 
'''

import os
import sys 
import time 
import json  
from pathlib import Path 
import pandas as pd 

import shutil
from parse_types import *
from parser_whisperjax import ParserWHPJax

from audio_split import split_audio_by_silence
from audio_export import gen_block_data, concact_block_files 

def get_file_info(file_path): 
    
    fileObj = FileObj() 
    fileObj.file_path = file_path


    p = Path(file_path)
    fileObj.file_extns = p.suffix[1:] 
    fileObj.file_name = p.name

    fileObj.work_dir = file_path.replace(f'.{fileObj.file_extns}', '') 

    if fileObj.file_extns in EXT_AUDIO:
        fileObj.file_type = FileType.audio
 
    return fileObj


class ParseAudioAdmin(object):

    def __init__(self, file_path, parse_engine=ParseEngine.whisper_jax ) -> None:
        
        self.fileObj = get_file_info(file_path) 
        self.parse_engine = parse_engine
         
        # 设置语音识别引擎 
        if self.parse_engine == ParseEngine.whisper_jax:
            import jax.numpy as jnp
            self.parser = ParserWHPJax(model_name="openai/whisper-large-v3", dtype=jnp.bfloat16, batch_size=16) 
            pass 


    # 静音切分后的数据 --> 识别
    def prcs_chunck_dir(self, chunks_dir, blocks_dir): 
        
        chunk_files = os.listdir(chunks_dir)  
        chunk_files.sort()   
        print('-- chunk_files : ', len(chunk_files) )
        # 静音切分后的数据，使用 whisper-jax 提取 timestamp、text   
        for sub_file_name in chunk_files:
            if not sub_file_name.endswith(self.fileObj.file_extns):continue
            
            sub_audio_file_path = os.path.join(chunks_dir, sub_file_name)   

            # 语音转文本 
            stt_ret = self.parser.parse_file(sub_audio_file_path) 
            print('-- stt_ret :', stt_ret)   
            
            stt_path = os.path.join(chunks_dir, sub_file_name.replace(f'.{self.fileObj.file_extns}', '.json'))
            with open(stt_path, 'w') as f:f.write(json.dumps(stt_ret, ensure_ascii=False )) 
     
        # tts 后，生成统一格式数据
        chunk_files = os.listdir(chunks_dir)
        chunk_files.sort()  
 
        audio_start_time = 0   # 本段开始时间
        start_block_id = 0 # 本段开始 id 
 
        for sub_file_name in chunk_files:
            if not sub_file_name.endswith('.json'):continue
            sub_stt_path = os.path.join(chunks_dir, sub_file_name)
            sub_audio_path = sub_stt_path.replace('.json', f'.{self.fileObj.file_extns}') 
            
            audio_start_time, start_block_id = gen_block_data(sub_audio_path, sub_stt_path, audio_start_time, start_block_id, blocks_dir, self.fileObj.file_extns)    
         

    def prcs_audio(self):
        if self.fileObj.file_type != FileType.audio:return 
          
        # 存储 根据静音 切分的数据块 
        chunks_dir = os.path.join(self.fileObj.work_dir, 'chunks') 
        if not os.path.exists(chunks_dir):os.makedirs(chunks_dir) 

        split_audio_by_silence(self.fileObj, chunks_dir=chunks_dir, min_silence_len=1000, silence_thresh=-70) 
 
        # 存储根据 whisper-jax 分块后的数据
        blocks_dir = os.path.join(self.fileObj.work_dir , 'blocks' ) 
        if not os.path.isdir(blocks_dir):os.makedirs(blocks_dir) 
 

        self.prcs_chunck_dir(chunks_dir, blocks_dir)          
        
        # 存储 block 按照大小限制 拼接的数据 
        ret_dir = os.path.join(self.fileObj.work_dir , 'ret' )  
        concact_block_files(blocks_dir, ret_dir)

        # 删除中间文件 
        shutil.rmtree(chunks_dir)
        shutil.rmtree(blocks_dir)
    
    
def process_audio_file(file_path):
    if not os.path.isfile(file_path):return
    try: 
        pa = ParseAudioAdmin(file_path, parse_engine=ParseEngine.whisper_jax)  
        pa.prcs_audio() 
    except Exception as err:
        print('xx', file_path, err) 
     

def handle_paths(paths):
    for path in paths: 
        if os.path.isfile(path):
            print('-- ', path) 
            process_audio_file(path)

        if os.path.isdir(path): 
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name) 
                process_audio_file(path) 
 

if __name__ == '__main__':
    
    paths = sys.argv[1:]
    # paths = ['/Users/xx/Documents/data/audio/1.mp3'] 
    print('-- ', paths) 
    handle_paths(paths) 
    




 