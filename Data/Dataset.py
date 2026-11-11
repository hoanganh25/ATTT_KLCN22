"""Minimal V0.1 saved float32 96x96 LibriSpeech patch dataset. No COALA/HDF5."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class SavedMelDataset(Dataset):
    def __init__(self,file_path):
        self.file_path=Path(file_path)
        if not self.file_path.is_file():raise FileNotFoundError(self.file_path)
        arr=np.load(self.file_path,mmap_mode='r',allow_pickle=False)
        if arr.ndim!=3 or arr.shape[1:]!=(96,96) or arr.dtype!=np.float32:
            raise ValueError(f'Expected float32 [N,96,96] in {self.file_path}; got {arr.shape} {arr.dtype}')
        self.count=len(arr)
        self._data=None
        del arr

    def __len__(self):return self.count

    def __getitem__(self,index):
        if self._data is None:self._data=np.load(self.file_path,mmap_mode='r',allow_pickle=False)
        x=np.array(self._data[index],dtype=np.float32,copy=True)
        if not np.isfinite(x).all():raise RuntimeError(f'Nonfinite dataset patch: {index}')
        return torch.from_numpy(x).unsqueeze(0)

    def __getstate__(self):
        state=self.__dict__.copy();state['_data']=None
        return state


def load_saved_loaders(batch_size=16,pin_memory=False,num_workers=0,seed=42,split_dir='Data/splits'):
    root=Path(split_dir)
    meta_file=root/'manifest.json'
    if not meta_file.is_file():raise FileNotFoundError(f'Missing prepared manifest: {meta_file}')
    meta=json.loads(meta_file.read_text(encoding='utf-8'))
    if meta.get('shape')!=[96,96]:raise ValueError('Manifest patch shape is not 96x96')
    ds={name:SavedMelDataset(root/f'{name}.npy') for name in ('train','val','test')}
    for name,part in ds.items():
        if len(part)!=meta['sizes'][name]:raise ValueError(f'Split size mismatch for {name}')
    g=torch.Generator();g.manual_seed(seed)
    train=DataLoader(ds['train'],batch_size=batch_size,shuffle=True,generator=g,
                     pin_memory=pin_memory,num_workers=num_workers)
    val=DataLoader(ds['val'],batch_size=batch_size,shuffle=False,pin_memory=pin_memory,num_workers=num_workers)
    test=DataLoader(ds['test'],batch_size=batch_size,shuffle=False,pin_memory=pin_memory,num_workers=num_workers)
    return train,val,test
