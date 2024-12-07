# mnbvc_parser
本 repo 目前主要处理 音频数据，输出为统一格式。主要过程在 `parse_audio.py` 文件。

你可以用如下方式来调用 ：

```python
from parse_audio import ParseAudioAdmin

file_path = '.../a.mp3' 
pa = ParseAudioAdmin(file_path, parse_engine=ParseEngine.whisper_jax)  
pa.prcs_audio() 

```



