import argparse
import os
import time
import torch
import torchaudio
import sys
import gdown

sys.path.append(os.path.join(os.path.dirname(__file__), "../languagecodec"))
from languagecodec_encoder.utils import convert_audio
from languagecodec_decoder.pretrained import Vocos

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../languagecodec/configs/languagecodec_mm.yaml")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../checkpoints/languagecodec_paper.ckpt")
SAMPLE_RATE = 24000


def load_audio(audio_file, target_sr=None, target_channels=1, device='cpu'):
    audio, sr = torchaudio.load(audio_file)

    # Mix to mono if necessary
    if audio.shape[0] > 1 and target_channels == 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Expand channels if necessary
    if audio.shape[0] == 1 and target_channels == 2:
        audio = audio.expand(target_channels, audio.shape[1])

    # Resample if necessary
    if target_sr is not None and sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    audio = audio.to(device)
    return audio, sr


def download_languagecodec_model(model_path):
    print("Model file not found. Downloading from Google Drive using gdown...")
    # Google Drive file ID and URL for the model file
    file_id = "1ENLyQzbJm2BTignliHHZl11DmQdWZAoX"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download the file using gdown
    gdown.download(url, model_path, quiet=False)
    if os.path.exists(model_path):
        print("Model downloaded successfully.")
    else:
        raise RuntimeError("Failed to download model.")


def load_languagecodec_model(config_path, model_path, device):
    if not os.path.exists(model_path):
        download_languagecodec_model(model_path)
    model = Vocos.from_pretrained0802(config_path, model_path)
    model.eval().to(device)
    return model


def run_languagecodec(model, audio, bandwidth_id):
    with torch.no_grad():
        # Encode the audio
        start_time = time.time()
        features, discrete_code = model.encode_infer(audio, bandwidth_id=bandwidth_id)
        processing_time = time.time() - start_time
        print(f"Encode Time: {processing_time:.4f} seconds")

        # Decode the audio
        start_time = time.time()
        reconstructed_audio = model.decode(features, bandwidth_id=bandwidth_id)
        processing_time = time.time() - start_time
        print(f"Decode Time: {processing_time:.4f} seconds")

    return reconstructed_audio


def test_languagecodec(model, test_set, device, target_sr):
    total_processing_time = 0
    total_input_duration = 0
    bandwidth_id = torch.tensor([0]).to(device)

    for audio_file, output_file in test_set:
        # Load the input audio and measure its duration
        audio, sr = load_audio(audio_file, target_sr=target_sr, target_channels=1, device=device)
        input_duration = audio.size(1) / sr
        total_input_duration += input_duration

        # Measure processing time
        start_time = time.time()
        reconstructed_audio = run_languagecodec(model, audio, bandwidth_id)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # Calculate RTF
        rtf = processing_time / input_duration

        # Save the output audio
        torchaudio.save(output_file, reconstructed_audio.cpu(), sr, encoding='PCM_S', bits_per_sample=16)

        # Print RTF for this audio file
        print(f"Audio file: {audio_file}")
        print(f"Processing Time: {processing_time:.4f} seconds")
        print(f"Input Duration: {input_duration:.4f} seconds")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Calculate average RTF
    average_rtf = total_processing_time / total_input_duration
    print(f"\nAverage Real-Time Factor (RTF): {average_rtf:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test Language-Codec model for audio reconstruction.")
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
    model = load_languagecodec_model(CONFIG_PATH, MODEL_PATH, device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    # Test model
    test_languagecodec(model, test_set, device, target_sr=SAMPLE_RATE)


if __name__ == '__main__':
    main()
