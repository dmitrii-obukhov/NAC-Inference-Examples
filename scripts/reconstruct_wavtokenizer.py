import argparse
import os
import time
import torch
import torchaudio
import sys
from huggingface_hub import hf_hub_download

sys.path.append(os.path.join(os.path.dirname(__file__), "../wavtokenizer"))
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer


SAMPLE_RATE = 24000
CONFIG_PATHS = {
    "WavTokenizer_small_600_24k_4096": "../wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    "WavTokenizer_small_320_24k_4096": "../wavtokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    "wavtokenizer_medium_speech_320_24k_v2": "../wavtokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
}
MODEL_REPO_IDS = {
    "WavTokenizer_small_600_24k_4096": "novateur/WavTokenizer",
    "WavTokenizer_small_320_24k_4096": "novateur/WavTokenizer",
    "wavtokenizer_medium_speech_320_24k_v2": "novateur/WavTokenizer-medium-speech-75token"
}
MODEL_FILENAMES = {
    "WavTokenizer_small_600_24k_4096": "WavTokenizer_small_600_24k_4096.ckpt",
    "WavTokenizer_small_320_24k_4096": "WavTokenizer_small_320_24k_4096.ckpt",
    "wavtokenizer_medium_speech_320_24k_v2": "wavtokenizer_medium_speech_320_24k_v2.ckpt"
}


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


def download_wavtokenizer_model(model_name, model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found. Downloading {model_name} from Hugging Face Hub...")
        repo_id = MODEL_REPO_IDS[model_name]
        filename = MODEL_FILENAMES[model_name]
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Model {model_name} downloaded successfully to {model_path}.")
    return model_path


def load_wavtokenizer_model(config_path, model_path, model_name, device):
    model_path = download_wavtokenizer_model(model_name, model_path)  # Ensure model is downloaded
    model = WavTokenizer.from_pretrained0802(config_path, model_path)
    model.eval().to(device)
    return model


def run_wavtokenizer(model, audio, bandwidth_id):
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


def test_wavtokenizer(model, test_set, device, target_sr):
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
        reconstructed_audio = run_wavtokenizer(model, audio, bandwidth_id)
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
    parser = argparse.ArgumentParser(description="Test WavTokenizer model for audio reconstruction.")
    parser.add_argument("--model_name", type=str, required=True, choices=CONFIG_PATHS.keys(),
                        help="Specify model name for WavTokenizer: 'WavTokenizer_small_600_24k_4096', 'WavTokenizer_small_320_24k_4096', or 'wavtokenizer_medium_speech_320_24k_v2'")
    parser.add_argument("--input_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, help="Path to save the output audio file")
    parser.add_argument("--input_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed audio files")

    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), CONFIG_PATHS[args.model_name])
    model_path = os.path.join(os.path.dirname(__file__), f"../checkpoints/{args.model_name}.ckpt")

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
    model = load_wavtokenizer_model(config_path, model_path, args.model_name, device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    # Test model
    test_wavtokenizer(model, test_set, device, target_sr=SAMPLE_RATE)


if __name__ == '__main__':
    main()
