import argparse
import os
import time
import torch
import torchaudio
import sys
import yaml
import librosa
import warnings

warnings.simplefilter('ignore')

# Add path for FAcodec Platchaa module
sys.path.append(os.path.join(os.path.dirname(__file__), "../facodec_platchaa"))
from modules.commons import *
from hf_utils import load_custom_model_from_hf
from losses import *


SAMPLE_RATE = 24000


def load_audio(audio_file, target_sr=SAMPLE_RATE, target_channels=1, device='cpu'):
    audio, sr = librosa.load(audio_file, sr=target_sr)
    audio = torch.tensor(audio).unsqueeze(0).float().to(device)

    # Mix to mono if necessary
    if audio.shape[0] > 1 and target_channels == 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Expand channels if necessary
    if audio.shape[0] == 1 and target_channels == 2:
        audio = audio.expand(target_channels, audio.shape[1])

    return audio, sr


def load_facodec_model(args, device):
    if not args.ckpt_path and not args.config_path:
        print("No checkpoint path or config path provided. Loading from huggingface model hub")
        ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
    else:
        print("Found existing checkpoint")
        ckpt_path = args.ckpt_path
        config_path = args.config_path
    
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params)

    ckpt_params = torch.load(ckpt_path, map_location="cpu")
    ckpt_params = ckpt_params['net'] if 'net' in ckpt_params else ckpt_params  # adapt to format of self-trained checkpoints

    for key in ckpt_params:
        model[key].load_state_dict(ckpt_params[key])

    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    return model


@torch.no_grad()
def run_facodec(model, source_audio, device):
    # Encode the audio
    start_time = time.time()
    z = model.encoder(source_audio[None, ...].to(device).float())
    processing_time = time.time() - start_time
    print(f"Encode Time: {processing_time:.4f} seconds")

    # Quantize the audio
    start_time = time.time()
    z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, source_audio[None, ...].to(device).float(), n_c=2)
    processing_time = time.time() - start_time
    print(f"Quantize Time: {processing_time:.4f} seconds")

    # Decode the audio
    start_time = time.time()
    full_pred_wave = model.decoder(z)
    processing_time = time.time() - start_time
    print(f"Decode Time: {processing_time:.4f} seconds")

    return full_pred_wave


def test_facodec(model, test_set, device, target_sr):
    total_processing_time = 0
    total_input_duration = 0

    for audio_file, output_file in test_set:
        # Load the input audio and measure its duration
        source_audio, sr = load_audio(audio_file, target_sr=target_sr, target_channels=1, device=device)
        input_duration = source_audio.size(1) / sr
        total_input_duration += input_duration

        # Measure processing time
        start_time = time.time()
        full_pred_wave = run_facodec(model, source_audio, device)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # Calculate RTF
        rtf = processing_time / input_duration

        # Save the output audio
        torchaudio.save(output_file, full_pred_wave[0].cpu(), sr, encoding='PCM_S', bits_per_sample=16)

        # Print RTF for this audio file
        print(f"Audio file: {audio_file}")
        print(f"Processing Time: {processing_time:.4f} seconds")
        print(f"Input Duration: {input_duration:.4f} seconds")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Calculate average RTF
    average_rtf = total_processing_time / total_input_duration
    print(f"\nAverage Real-Time Factor (RTF): {average_rtf:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test FAcodec Platchaa model for audio reconstruction.")
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to the model checkpoint.")
    parser.add_argument("--config_path", type=str, default="", help="Path to the model config file.")
    parser.add_argument("--input_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, help="Path to save the output audio file")
    parser.add_argument("--input_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed audio files")

    args = parser.parse_args()

    if (args.input_file and args.output_file) and not (args.input_dir or args.output_dir):
        # Single file processing
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        test_set = [(args.input_file, args.output_file)]
    elif (args.input_dir and args.output_dir) and not (args.input_file or args.output_file):
        # Directory processing
        os.makedirs(args.output_dir, exist_ok=True)
        test_set = [
            (
                os.path.join(args.input_dir, audio_file),
                os.path.join(args.output_dir, audio_file)
            )
            for audio_file in os.listdir(args.input_dir)
            if audio_file.endswith('.wav')
        ]
    else:
        parser.error("Specify either input_file/output_file or input_dir/output_dir, not both.")

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    start_time = time.time()
    model = load_facodec_model(args, device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    # Test model
    test_facodec(model, test_set, device, target_sr=SAMPLE_RATE)


if __name__ == '__main__':
    main()
