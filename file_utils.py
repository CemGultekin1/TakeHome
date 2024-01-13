
import numpy as np
import os
import json

class NumpyArrayDictionary(dict):
    dotnpy :str = '.npy'
    def __init__(self,):
        super().__init__()
    def save_to_file(self,path:str,):
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        for filename,value in self.items():
            filepath = os.path.join(path,'_'+str(filename))
            np.save(filepath,value)
    def load_from_file(self,path:str):
        if not os.path.exists(path):
            return        
        npyfiles = [(f.replace(self.dotnpy,''),f) for f in os.listdir(path) if self.dotnpy in f]
        for varname,filename in npyfiles:
            arr = np.load(os.path.join(path,filename))
            self[varname[1:]] = arr

class MixedTypeStorage(NumpyArrayDictionary):
    json_filename:str = 'meta.json'
    mts_extension:str = '.mts'
    def __init__(self) -> None:
        super().__init__()
        self.json = {}
    def __setitem__(self,name:str,value):
        if isinstance(value,np.ndarray):
            super().__setitem__(name,value)
            return
        self.json[name] = value
    def __getitem__(self,key):
        if key in  self.json:
            return self.json[key]
        return super().__getitem__(key)
    def extension_correction(self,path :str ):
        return path.replace(self.mts_extension,'') + self.mts_extension
    def save_to_file(self, path: str):
        path = self.extension_correction(path)
        print(f'saving to folder = {path}')
        super().save_to_file(path)
        with open(os.path.join(path,self.json_filename), 'w') as outfile:
            json.dump(self.json,outfile)
    @staticmethod
    def from_dict(kwargs):
        x = MixedTypeStorage()
        for key,val in kwargs.items():
            x[key] = val
        return x
    def load_from_file(self, path: str):
        path = self.extension_correction(path)
        super().load_from_file(path)
        with open(os.path.join(path,self.json_filename), 'r') as openfile:
            self.json = json.load(openfile)
        