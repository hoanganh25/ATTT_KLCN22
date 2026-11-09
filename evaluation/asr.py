"""Optional Whisper ASR inference for WER/CER (downloads model on first use)."""
import numpy as np


class SpeechRecognizer:
    def __init__(self,model_name='small.en',device='cpu'):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('WER/CER requested but faster-whisper not installed. pip install -r requirements_asr.txt') from exc
        self.model_name=model_name
        self.model=WhisperModel(model_name,device=device,compute_type='float32' if device=='cpu' else 'float16')

    def transcribe(self,waveform,sample_rate=16000):
        if sample_rate!=16000:raise ValueError('Whisper expects 16 kHz mono')
        x=np.asarray(waveform,dtype=np.float32).ravel()
        segments,_=self.model.transcribe(x,language='en',beam_size=5,condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segments).strip()
