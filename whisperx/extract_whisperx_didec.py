import whisperx
import os
import json
import torch
import numpy as np
from whisper.audio import load_audio
import soundfile as sf

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = "cuda"

# load alignment model and metadata
model_a, metadata = whisperx.load_align_model(language_code='nl', device=device)
# jonatasgrosman/wav2vec2-large-xlsr-53-dutch

if not os.path.exists('alignments_whisperX'):
    os.mkdir('alignments_whisperX')

count_f = 0

for root, subdirs, files in os.walk('converted_wavs/'):

    for f in files:

        count_f += 1
        if count_f % 10 == 0:
            print(count_f)

        _, _, ppn, img = f.split('.')[0].split('_')
        ppn = ppn.split('ppn')[1]

        grammar_path = 'grammars/'
        with open(grammar_path + 'raw_caption_' + ppn + '_' + img + '.jsgf', 'r') as g:
            lines = g.readlines()
            for line in lines:
                if 'public <s> = ' in line:
                    utterance = line.split('public <s> = ')[1].split(';')[0]

        audio_file = os.path.join(root, f)
        audio = load_audio(audio_file)
        audio = torch.from_numpy(audio)
        duration = audio.shape[0] / 16000

        # checked this previously
        # f = sf.SoundFile(audio_file)
        # print('samples = {}'.format(f.frames))
        # print('sample rate = {}'.format(f.samplerate))
        # print('seconds = {}'.format(f.frames / f.samplerate))
        # assert f.samplerate == 16000
        # assert f.frames / f.samplerate == duration

        segments = [{'text': utterance, 'start': 0.0, 'end': duration}]

        # align whisper output
        result_aligned = whisperx.align(segments, model_a, metadata, audio_file, device)

        # print(result_aligned["segments"]) # after alignment
        # print(result_aligned["word_segments"]) # after alignment

        with open('alignments_whisperX/aligned_whisperx_' + ppn + '_' + img + '.json', 'w') as wa:
            json.dump(result_aligned["word_segments"], wa)

print('count', count_f)
