

import argparse
import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import librosa
import pandas as pd
from datasets import load_dataset
from transformers import Wav2Vec2Processor
import torchaudioimport os
import sys
import torchaudio
import librosa
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import argparse

def load_audio_and_resample(batch):
    """
    Load audio files and resample them to match the model's expected sampling rate.

    :param batch: Dict with the data including the 'path' key.
    """
    audio, orig_sampling_rate = torchaudio.load(batch['path'])
    audio = audio.squeeze().numpy()
    resampled_audio = librosa.resample(audio, orig_sampling_rate, TARGET_SAMPLING_RATE)

    batch['audio'] = resampled_audio
    return batch

def feature_extraction(batch, model, processor, device):
   
    inputs = processor(batch['audio'], sampling_rate=TARGET_SAMPLING_RATE, return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
    
    batch['features'] = outputs.last_hidden_state
    return batch

def main():
    parser = argparse.ArgumentParser(description="Extract features from audio data using Wav2Vec model.")
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to the dataset (CSV file with audio paths).')
    parser.add_argument('-out', '--out_dir', type=str, default='./', help='Output directory to save the features.')
    parser.add_argument('-model', '--model_id', type=str, default='jonatasgrosman/wav2vec2-large-xlsr-53-english', help='Wav2Vec model ID.')

    args = parser.parse_args()

    global TARGET_SAMPLING_RATE
    TARGET_SAMPLING_RATE = 16000  # Set the target sampling rate

    # Load dataset
    dataset = load_dataset('csv', data_files={'test': os.path.join(args.data, 'train.csv')}, delimiter='\t')['test']

    # Set up model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2Model.from_pretrained(args.model_id).to(device)

    # Process dataset
    dataset = dataset.map(load_audio_and_resample)
    dataset = dataset.map(lambda batch: feature_extraction(batch, model, processor, device))

    # Save features to CSV
    feature_columns = ['feature_' + str(i) for i in range(1024)]  
    for record in dataset:
        features_df = pd.DataFrame(record['features'].cpu().numpy(), columns=feature_columns)
        features_df.insert(0, 'file_path', record['path'])  # Add file path as the first column
        features_df.to_csv(os.path.join(args.out_dir, os.path.basename(record['path']) + '.csv'), sep=';', index=False)

if __name__ == '__main__':
    main()

import numpy as np
import torch
from src.Audio.FineTuningWav2Vec.Wav2VecAuxClasses import *

def speech_file_to_array_fn(batch):
   
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def extract_features(batch, device, model, processor):
    """
        Generate features from the model and append to the batch dict the posteriors and predictions

        :param batch:[dict] Dict with the data
                                -speech: input audio recordings [IN]
                                -predicted : Embeddings extracted from the last layer of the feature encoder (CNNs block)
        :param: device [str]: Device used to load the model and make predictions ('cpu' or 'cuda')
        :param: model [Wav2Vec2Model]: Model to extract the embeddings from.
        :param processor[Wav2Vec2Processor]: Information of the expected input format of the model

    """
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)
        #feats = processor.feature_extractor(input_values)

    batch["predicted"] = logits["extract_features"]
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path with the datasets automatically generated with the Fine-Tuning script (train.csv and test.csv)')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the embeddings extracted from the model', default='./')
    parser.add_argument('-model', '--model_id', type=str, help='Model identificator in Hugging Face library [default: jonatasgrosman/wav2vec2-large-xlsr-53-english]',
                        default='jonatasgrosman/wav2vec2-large-xlsr-53-english')

    args = parser.parse_args()

    test_dataset = load_dataset("csv", data_files={"test": os.path.join(args.data, "train.csv")}, delimiter="\t")["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2Model.from_pretrained(args.model_id).to(device)
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    cols = ["embs" + str(i) for i in range(512)]
    for row in test_dataset:
        result = extract_features(row, device, model, processor)
        df_aux = pd.DataFrame(result['predicted'].cpu().numpy().reshape(-1, 512), columns=cols)
        df_aux.to_csv(os.path.join(args.out_dir, result["name"]+".csv"), sep=";", index=False, header=True)






