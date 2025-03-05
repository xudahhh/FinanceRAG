#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir="./embedding/")