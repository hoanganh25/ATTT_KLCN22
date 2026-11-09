"""Official V0.1 test evaluator. Modes: mel (whole-utterance log-Mel MSE), wave (full-utterance MSE/PESQ/STOI).
Supports CNN Baseline and adapted SE-ResNet, AWGN/Rayleigh/no_noise.
Waveform synthesis uses only the Power-Mel Joint implementation in mel_inverse.
No ASR, WER/CER or diagnostic fields in formal CSVs.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from channels.factory import make_channel
from evaluation.stoi import score_stoi
from evaluation.pesq import score_pesq
from evaluation.waveform_mse import score_waveform_mse
from mel_frontend.audio_io import read_wave
from mel_frontend.config import MelConfig
from mel_inverse.receiver import recover_full_logmel
from mel_inverse.power_joint import synthesize


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['mel','wave','both'],default='both')
    p.add_argument('--energy-mask-threshold',type=float,default=None,
        help='Optional experimental energy-mask Precision/Recall/F1: raw ln(Mel-power) threshold, must be predeclared using TRAIN/DEV only. NOT a speech classification metric.')
    p.add_argument('--splits',default='Data/splits')
    p.add_argument('--checkpoint',action='append',required=True,metavar='NAME=PATH')
    p.add_argument('--architecture',action='append',default=[],metavar='NAME=cnn|se_resnet',help='Per-checkpoint architecture; auto inference when omitted')
    p.add_argument('--channels',default='awgn')
    p.add_argument('--snrs',default='0,5,10,15,20')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--max-utterances',type=int,default=30,help='0=all test utterances')
    p.add_argument('--iterations',default='64',help='Power-Mel Joint iterations; final protocol defaults to 64')
    p.add_argument('--joint-device',choices=['auto','cuda','cpu','torch-cpu'],default='auto')
    p.add_argument('--precision',choices=['float32','float64'],default='float32')
    p.add_argument('--lambda-mel',type=float,default=1.0)
    p.add_argument('--rho',type=float,default=1.0)
    p.add_argument('--inner-steps',type=int,default=5)
    p.add_argument('--init-iterations',type=int,default=40)
    p.add_argument('--init',choices=['multiplicative','nnls'],default='multiplicative')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu',action='store_true',help='CNN on CPU; joint device controlled separately')
    p.add_argument('--out',default='reports/final_joint')
    return p.parse_args()


def parse_pairs(values,field):
    out={}
    for item in values:
        if '=' not in item:raise ValueError(f'{field}: use NAME=VALUE')
        name,value=item.split('=',1)
        if not name or not value or name in out:raise ValueError(f'{field}: missing/duplicate name {item}')
        out[name]=value
    return out


def csv_save(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def import_se_model():
    # Folder name SE-ResNet is not valid in a Python `from ... import ...` statement.
    # Register a synthetic package so relative .Encoder import in Decode.py works.
    folder=Path('model/SE-ResNet').resolve()
    if not folder.is_dir():raise FileNotFoundError(f'Missing {folder}')
    module_name='_v01_se_resnet'
    import types
    pkg=types.ModuleType(module_name);pkg.__path__=[str(folder)]
    sys.modules[module_name]=pkg
    def load(short):
        spec=importlib.util.spec_from_file_location(f'{module_name}.{short}',folder/f'{short}.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        return module
    return load('Encoder'),load('Decode')


def load_checkpoint(path,architecture,split_hash,device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    sig=ck.get('data_manifest_sha256',ck.get('manifest_sha256'))
    if sig is not None and sig!=split_hash:raise RuntimeError(f'Checkpoint manifest differs: {path}')
    kind=architecture or ('se_resnet' if ck.get('model_config') or 'SE-' in str(ck.get('architecture','')) else 'cnn')
    if kind=='se_resnet':
        from dataclasses import fields
        enc_mod,dec_mod=import_se_model()
        config=enc_mod.SEConfig(**{f.name:ck.get('model_config',{}).get(f.name,f.default) for f in fields(enc_mod.SEConfig)})
        enc,dec=enc_mod.Encoder(config),dec_mod.Decoder(config)
    elif kind=='cnn':
        # Prefer new folder structure, but allow original V0.1 folder as fallback.
        if Path('model/CNN_Baseline/Encoder.py').exists():
            folder=Path('model/CNN_Baseline').resolve()
            import types
            pkg=types.ModuleType('_v01_cnn_baseline');pkg.__path__=[str(folder)]
            sys.modules[pkg.__name__]=pkg
            def load_module(name):
                spec=importlib.util.spec_from_file_location(f'_v01_cnn_baseline.{name}',folder/f'{name}.py')
                obj=importlib.util.module_from_spec(spec);sys.modules[spec.name]=obj;spec.loader.exec_module(obj)
                return obj
            enc,dec=load_module('Encoder').Encoder(),load_module('Decode').Decoder()
        else:
            from model.CNN_Baseline.Encoder import Encoder
            from model.CNN_Baseline.Decode import Decoder
            enc,dec=Encoder(),Decoder()
    else:raise ValueError(f'Unknown architecture {kind}')
    enc.load_state_dict(ck['encoder'],strict=True)
    dec.load_state_dict(ck['decoder'],strict=True)
    return enc.to(device).eval(),dec.to(device).eval(),ck,kind


def summary(rows,metrics,group_fields):
    grouped={}
    for row in rows:
        key=tuple(row[g] for g in group_fields)
        grouped.setdefault(key,[]).append(row)
    result=[]
    for key,records in grouped.items():
        out=dict(zip(group_fields,key));out['n']=len(records)
        for metric in metrics:
            vals=[float(r[metric]) for r in records if r.get(metric) is not None and np.isfinite(float(r[metric]))]
            out[metric]=float(np.mean(vals)) if vals else None
        result.append(out)
    return result


def energy_mask_scores(reference_logmel, predicted_logmel, threshold):
    """Experimental binary time-frequency energy mask (NOT a primary speech metric).

    Positive iff raw natural-log Mel POWER > predeclared threshold.
    Aggregates TP/FP/FN over all valid time-frequency cells in one utterance.
    Undefined precision/recall/F1 are NaN (not silently called 0 or 1).
    """
    ref=np.asarray(reference_logmel,dtype=np.float64)
    pred=np.asarray(predicted_logmel,dtype=np.float64)
    if ref.shape != pred.shape or ref.ndim != 2 or not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError('Energy mask requires same-shape finite 2D log-Mel arrays')
    gt=ref > threshold
    estimated=pred > threshold
    tp=int(np.count_nonzero(gt & estimated))
    fp=int(np.count_nonzero(~gt & estimated))
    fn=int(np.count_nonzero(gt & ~estimated))
    precision=tp/(tp+fp) if (tp+fp)>0 else float('nan')
    recall=tp/(tp+fn) if (tp+fn)>0 else float('nan')
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else float('nan')
    return dict(energy_mask_precision=precision,energy_mask_recall=recall,energy_mask_f1=f1)


def execute(a):
    if a.repeats<1 or a.batch_size<1 or a.max_utterances<0:raise ValueError('Invalid repeat/batch/maximum')
    if a.energy_mask_threshold is not None:
        if not np.isfinite(a.energy_mask_threshold):raise ValueError('Energy-mask threshold must be finite')
        if a.mode=='wave':raise ValueError('Energy-mask scores require --mode mel or both')
    checkpoints=parse_pairs(a.checkpoint,'checkpoint')
    arch=parse_pairs(a.architecture,'architecture')
    if set(arch)-set(checkpoints):raise ValueError('Architecture name missing checkpoint')
    channels=[x.strip() for x in a.channels.split(',') if x.strip()]
    if not channels or len(set(channels))!=len(channels) or set(channels)-{'awgn','rayleigh','no_noise'}:raise ValueError('Invalid channels')
    snrs=[float(s) for s in a.snrs.split(',')]
    if not snrs or not all(np.isfinite(snrs)):raise ValueError('Invalid SNR list')
    iters=sorted({int(s) for s in a.iterations.split(',')})
    if min(iters)<1:raise ValueError('Iterations must >=1')
    modes=[('no_noise',None)] if channels==['no_noise'] else [(c,None) if c=='no_noise' else (c,snr) for c in channels for snr in ([None] if c=='no_noise' else snrs)]
    source=Path(a.splits)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    normal=json.loads((source/'normalization.json').read_text(encoding='utf-8'))
    manifest_hash=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    mean=np.asarray(normal['mean'],np.float32);std=np.asarray(normal['std'],np.float32)
    if mean.shape!=(96,) or std.shape!=(96,):raise RuntimeError('Expected 96-bin normalization')
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    loaded={name:load_checkpoint(path,arch.get(name),manifest_hash,device) for name,path in checkpoints.items()}
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Output already contains files: {out}; choose a NEW --out to avoid mixing protocols')
    out.mkdir(parents=True,exist_ok=True)
    cfg=MelConfig(**manifest['dsp'])
    metadata={'mode':a.mode,'dataset':'test','checkpoints':{name:str(Path(p).resolve()) for name,p in checkpoints.items()},
              'architectures':{name:v[3] for name,v in loaded.items()},'manifest_sha256':manifest_hash,
              'channels':channels,'snrs':snrs,'repeats':a.repeats,'seed':a.seed,'batch_size':a.batch_size,
              'max_utterances':a.max_utterances,'experimental_energy_mask_threshold_raw_logmel':a.energy_mask_threshold,
              'synthesis':'power_mel_joint','joint_device':a.joint_device,'precision':a.precision,
              'iterations':iters,'lambda_mel':a.lambda_mel,'rho':a.rho,
              'inner_steps':a.inner_steps,'init_iterations':a.init_iterations,'init':a.init,
              'note':'Only normalized/raw logMel MSE, waveform MSE, STOI, PESQ-WB. No WER/CER/ASR/diagnostic metrics.'}
    (out/'evaluation_config.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    # One fixed selection for both modes, allowing independent MEL/WAVE invocations.
    # The waveform stage uses precisely the recovered log-Mel measured in the MEL stage.
    from evaluation.common import read_jsonl,choose_utterances
    cache=Path(manifest['source_cache_dir'])
    audio_root=Path(manifest['source_dataset_root'])
    items=choose_utterances(read_jsonl(cache/'test.jsonl'),a.max_utterances,a.seed)
    chosen=[{'file':it['relpath'],'speaker':it.get('speaker',''),'frames':int(it['frames'])} for it in items]
    (out/'selected_test_utterances.json').write_text(
        json.dumps(chosen,indent=2,ensure_ascii=False),encoding='utf-8')
    metadata['selected_utterances'] = len(chosen)
    metadata['mse_population'] = 'whole_utterance_frames_equal_weight_per_utterance'
    (out/'evaluation_config.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')

    do_mel=a.mode in ('mel','both')
    do_wave=a.mode in ('wave','both')
    mel_rows=[]
    wave_rows=[]
    extra_mask_fields=['energy_mask_precision','energy_mask_recall','energy_mask_f1'] if a.energy_mask_threshold is not None else []
    mel_fields=['file','speaker','model','channel','snr_db','repeat','norm_mse','raw_logmel_mse']+extra_mask_fields
    wave_fields=['file','speaker','model','channel','snr_db','repeat','iterations','raw_logmel_mse','waveform_mse','stoi','pesq_wb','runtime_s']
    mel_handle=wave_handle=None
    try:
        if do_mel:
            mel_handle=(out/'mel_detail.csv').open('w',newline='',encoding='utf-8-sig')
            mel_writer=csv.DictWriter(mel_handle,fieldnames=mel_fields)
            mel_writer.writeheader();mel_handle.flush()
        if do_wave:
            wave_handle=(out/'wave_detail.csv').open('w',newline='',encoding='utf-8-sig')
            wave_writer=csv.DictWriter(wave_handle,fieldnames=wave_fields)
            wave_writer.writeheader();wave_handle.flush()
        for idx,item in enumerate(items,1):
            mel=np.load(cache/item['cache'],allow_pickle=False)
            if mel.shape!=(96,int(item['frames'])):
                raise ValueError(f'Unexpected mel shape: {item["relpath"]}')
            reference=read_wave(audio_root/item['relpath'],cfg) if do_wave else None
            itemseed=(int(hashlib.sha256(item['relpath'].encode()).hexdigest()[:8],16)+a.seed)%(2**31)
            for name,(enc,dec,_,_) in loaded.items():
                for chname,snr in modes:
                    for rep in range(1 if chname=='no_noise' else a.repeats):
                        seed=itemseed+rep*100000+(0 if snr is None else int(snr*100))+{'no_noise':0,'awgn':10000,'rayleigh':20000}[chname]
                        torch.manual_seed(seed)
                        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
                        channel=make_channel(chname,10.0 if snr is None else snr).to(device)
                        recovered=recover_full_logmel(mel,enc,dec,channel,device,mean,std,batch_size=a.batch_size)
                        difference=recovered.astype(np.float64)-mel.astype(np.float64)
                        raw_mse=float(np.mean(difference**2))
                        norm_mse=float(np.mean((difference/std.astype(np.float64)[:,None])**2))
                        common=dict(file=item['relpath'],speaker=item.get('speaker',''),model=name,channel=chname,
                            snr_db='none' if snr is None else snr,repeat=rep+1)
                        if do_mel:
                            mel_row=dict(**common,norm_mse=norm_mse,raw_logmel_mse=raw_mse)
                            if a.energy_mask_threshold is not None:
                                mel_row.update(energy_mask_scores(mel,recovered,a.energy_mask_threshold))
                            mel_rows.append(mel_row)
                            mel_writer.writerow(mel_row);mel_handle.flush()
                            print(f'[MEL] {idx}/{len(items)} {name} {chname} {snr} MSE={norm_mse:.6f}',flush=True)
                        if do_wave:
                            waves=synthesize(recovered,cfg,len(reference),iterations=iters,device=a.joint_device,precision=a.precision,
                                lambda_mel=a.lambda_mel,rho=a.rho,inner_steps=a.inner_steps,
                                init_iterations=a.init_iterations,init=a.init,seed=itemseed)
                            for n,(wave,details) in waves.items():
                                wave_mse=score_waveform_mse(reference,wave)['waveform_mse']
                                wave_row=dict(**common,iterations=n,raw_logmel_mse=raw_mse,waveform_mse=wave_mse,
                                    stoi=score_stoi(reference,wave,cfg.sr),pesq_wb=score_pesq(reference,wave,cfg.sr),
                                    runtime_s=details.get('time_s'))
                                wave_rows.append(wave_row)
                                wave_writer.writerow(wave_row);wave_handle.flush()
                                print(f'[WAVE] {idx}/{len(items)} {name} {chname} {snr} iter={n} PESQ={wave_row["pesq_wb"]:.4f} STOI={wave_row["stoi"]:.4f}',flush=True)
    finally:
        if mel_handle is not None:mel_handle.close()
        if wave_handle is not None:wave_handle.close()
    if do_mel:
        group=['model','channel','snr_db'];met=['norm_mse','raw_logmel_mse']+extra_mask_fields
        csv_save(out/'mel_summary.csv',summary(mel_rows,met,group),group+['n']+met)
    if do_wave:
        group=['model','channel','snr_db','iterations'];met=['raw_logmel_mse','waveform_mse','stoi','pesq_wb','runtime_s']
        csv_save(out/'wave_summary.csv',summary(wave_rows,met,group),group+['n']+met)
    print('[DONE]',out.resolve(),flush=True)


if __name__=='__main__':execute(arguments())
