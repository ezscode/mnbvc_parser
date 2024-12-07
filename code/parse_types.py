#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parse_types.py
@Time    :   2024-12-07 14:21:12
@Author  :   Ez
@Version :   1.0
@Desc    :   数据类型

'''



from enum import StrEnum 

# 文件扩展名 和 所属分类  
EXT_AUDIO = ['mp3', 'mov', 'wav', 'ogg', 'flac', 'm4a']  
EXT_VIDEO = ['mp4', 'mepg']  
# 文件类型 - 文件内容信息 
class FileType(StrEnum): 
    audio = 'audio'  
    video = 'video'   

# 解析引擎 
class ParseEngine(StrEnum):
    whisper_jax = 'whisper_jax' 
 
# 数据类型 
class FileObj(object):

    def __init__(self) -> None:
        self.file_type = FileType.none 
        self.file_path = '' 
        self.file_name = ''
        self.file_id = '' 
        self.file_extns = '' # 文件名后缀
        self.work_dir = ''

        self.source = '' 
        self.url = ''
        self.title = ''
        self.content = ''
        self.addinfo = {} # 额外信息 

