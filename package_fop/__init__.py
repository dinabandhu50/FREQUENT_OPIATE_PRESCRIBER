import config 

with open(config.VERSION_PATH,'r') as fp:
    __version__ = fp.read().strip()