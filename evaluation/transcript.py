"""Find official LibriSpeech transcript aligned by utterance ID, without ASR inference."""
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=128)
def read_chapter_text(path: str):
    content={}
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split(maxsplit=1)
            if len(parts)==2:content[parts[0]]=parts[1]
    return content


def official_transcript(audio_path):
    audio_path=Path(audio_path)
    parts=audio_path.stem.split('-')
    if len(parts)<3:
        raise ValueError(f'Expected LibriSpeech speaker-chapter-utterance audio filename: {audio_path}')
    chapter=audio_path.parent/f'{parts[0]}-{parts[1]}.trans.txt'
    if not chapter.is_file():
        raise FileNotFoundError(f'Missing LibriSpeech transcript: {chapter}. Re-extract .trans.txt files with audio.')
    mapping=read_chapter_text(str(chapter))
    if audio_path.stem not in mapping:
        raise KeyError(f'No transcript for {audio_path.stem} in {chapter}')
    return mapping[audio_path.stem]
