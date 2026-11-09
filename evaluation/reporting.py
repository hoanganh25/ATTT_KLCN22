"""ONLY report presentation: aggregate per-utterance rows and generate figures/JSON/CSV."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import write_csv


def aggregate_speech(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['channel'],str(r['snr_db']))].append(r)
    out=[]
    for (model,channel,snr),group in sorted(groups.items()):
        record={'model':model,'channel':channel,'snr_db':snr,'utterances':len(group)}
        for metric in ('stoi','pesq_wb','wer','cer','wer_original_asr','cer_original_asr','utterance_raw_mse','clean_stoi','clean_pesq_wb'):
            vals=[]
            for item in group:
                v=item.get(metric)
                if isinstance(v,(float,int)) and np.isfinite(v):vals.append(float(v))
            record[f'{metric}_mean']=float(np.mean(vals)) if vals else None
            record[f'{metric}_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            record[f'{metric}_valid']=len(vals)
        # Waveform metrics are optional to preserve the exact schema of legacy runs
        # resumed with their original --metrics settings.
        if any('waveform_mse' in item for item in group):
            for metric in ('waveform_mse','waveform_rmse','waveform_nmse',
                           'reference_mean_power','clean_waveform_mse',
                           'clean_waveform_rmse','clean_waveform_nmse',
                           'recovered_vs_clean_waveform_mse'):
                vals=[float(item[metric]) for item in group
                      if item.get(metric) is not None
                      and isinstance(item.get(metric),(float,int))
                      and np.isfinite(item[metric])]
                record[f'{metric}_mean']=float(np.mean(vals)) if vals else None
                record[f'{metric}_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                record[f'{metric}_valid']=len(vals)
        # Micro-aggregated global error rate: better than mean per-utterance rates.
        for name in ('wer','cer'):
            e=sum(int(r.get(name+'_errors') or 0) for r in group)
            n=sum(int(r.get(name+'_ref_units') or 0) for r in group)
            record[f'{name}_micro']=float(e/n) if n else None
        out.append(record)
    return out


def save_reports(out_dir,mse_rows,speech_rows,metadata):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'mse_all_test_patches.csv',mse_rows)
    write_csv(out/'speech_metrics_detail.csv',speech_rows)
    summary=aggregate_speech(speech_rows)
    write_csv(out/'speech_metrics_summary.csv',summary)
    (out/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,items,key,title in [('mse',mse_rows,'raw_logmel_mse','Raw log-Mel MSE (lower better)'),
                                 ('stoi',summary,'stoi_mean','STOI (higher better)'),
                                 ('pesq',summary,'pesq_wb_mean','PESQ-WB MOS-LQO (higher better)'),
                                 ('wer',summary,'wer_micro','WER (lower better)'),
                                 ('cer',summary,'cer_micro','CER (lower better)'),
                                 ('waveform_mse',summary,'waveform_mse_mean','Waveform MSE (lower better)'),
                                 ('waveform_nmse',summary,'waveform_nmse_mean','Waveform NMSE (lower better)')]:
        fig,ax=plt.subplots(figsize=(8,5));anyline=False
        cats=sorted({(r['model'],r['channel']) for r in items})
        for model,channel in cats:
            data=[r for r in items if r['model']==model and r['channel']==channel and r['snr_db']!='none'
                  and isinstance(r.get(key),(float,int)) and np.isfinite(r[key])]
            if data:
                data.sort(key=lambda r:float(r['snr_db']))
                ax.plot([float(r['snr_db']) for r in data],[r[key] for r in data],marker='o',label=f'{model} / {channel}')
                anyline=True
        if anyline:
            ax.set(xlabel='Nominal SNR (dB)',ylabel=title,title=title)
            ax.grid(alpha=.25);ax.legend()
            fig.tight_layout();fig.savefig(out/f'{name}_vs_snr.png',dpi=150)
        plt.close(fig)
    return summary
