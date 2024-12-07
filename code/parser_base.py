#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   parser_base.py
@Time    :   2024-12-07 14:21:26
@Author  :   Ez
@Version :   1.0
@Desc    :   处理器 

'''



import os 
from abc import abstractmethod 

class Parser(object):

    def __init__(self, *args, **kwargs): 
        pass 
         
    def parse_file(self, file_path): 
        pass         
 
    def parse_url(self, url): 
        pass         
