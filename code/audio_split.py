#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parser_silence.py
@Time    :   2024-12-07 11:55:14
@Author  :   Ez
@Version :   1.0
@Desc    :   根据静音切分音频 

'''

import os  
from parse_types import *
from parser_base import Parser


from pydub  import AudioSegment
from pydub.silence import split_on_silence 
 
    
def split_audio_by_silence(fileObj, chunks_dir='', min_silence_len=1000, silence_thresh=-70):
    """
    min_silence_len: 拆分语句时，静默满0.3秒则拆分
    silence_thresh：小于-70dBFS以下的为静默
    """
    sound = AudioSegment.from_file(fileObj.file_path, format=fileObj.file_extns)
    # 分割 
    print('\n---- start split by silence', fileObj.file_path)
    chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh) 
    
    if len(chunks_dir) > 0 and len(chunks) > 0: 
        # 保存所有分段 
        for i, chunk in enumerate(chunks):
            save_path = os.path.join(chunks_dir, f'{i:04d}.{fileObj.file_extns}') 
            print(f'-- {i:04d}',  len(chunk), save_path )
            # print('-- ', save_path)
            chunk.export(save_path, format=fileObj.file_extns)

    print('== end split by silence ', len(chunks))
