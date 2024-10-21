import argparse
import os
import time
import torch
import torchaudio

from vocos import Vocos


SAMPLE_RATE = 24000


def load_audio(audio_file, target_sr=None, device='cpu'):
    audio, sr = torchaudio.load(audio_file)
    
    # Mix to mono if necessary
    if audio.shape[0] > 1: 
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if necessary
    if target_sr is not None and sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    audio = audio.to(device)
    return audio, sr


def load_vocos(features, device):
    if features == "mel":
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    elif features == "encodec":
        vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
    
    vocos.to(device)
    return vocos


def run_vocos(vocos, audio, features, bandwidth=None):
    if features == "mel":
        return vocos(audio)
    elif features == "encodec":
        # Map bandwidth values to their corresponding bandwidth_id
        bandwidth_map = {
            1.5: torch.tensor([0]),
            3.0: torch.tensor([1]),
            6.0: torch.tensor([2]),
            12.0: torch.tensor([3])
        }
        bandwidth_id = bandwidth_map.get(bandwidth).to(audio.device)
        return vocos(audio, bandwidth_id=bandwidth_id)
    

def test_vocos(vocos, test_set, features, bandwidth, device):
    total_processing_time = 0
    total_input_duration = 0
    rtf_values = []

    for audio_file, output_file in test_set:
        # Load the input audio and measure its duration
        audio, sr = load_audio(audio_file, target_sr=SAMPLE_RATE, device=device)
        input_duration = audio.size(1) / sr
        total_input_duration += input_duration

        # Measure processing time
        start_time = time.time()
        audio_reconstructed = run_vocos(vocos, audio, features, bandwidth)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # Calculate RTF
        rtf = processing_time / input_duration
        rtf_values.append(rtf)

        # Save the output audio
        torchaudio.save(output_file, audio_reconstructed.cpu(), sr)

        # Print RTF for this audio file
        print(f"Audio file: {audio_file}")
        print(f"Processing Time: {processing_time:.4f} seconds")
        print(f"Input Duration: {input_duration:.4f} seconds")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Calculate average RTF
    average_rtf = total_processing_time / total_input_duration
    print(f"\nAverage Real-Time Factor (RTF): {average_rtf:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test Vocos model with specified features.")
    parser.add_argument("--features", type=str, required=True, choices=["mel", "encodec"],
                        help="Choose the feature type for Vocos: 'mel' or 'encodec'")
    parser.add_argument("--bandwidth", type=float, choices=[1.5, 3.0, 6.0, 12.0],
                        help="Specify bandwidth for 'encodec' features. Choices: 1.5, 3.0, 6.0, 12.0")
    parser.add_argument("--input_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, help="Path to save the output audio file")
    parser.add_argument("--input_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed audio files")
      
    args = parser.parse_args()
    if args.features == "encodec" and args.bandwidth is None:
        parser.error("--bandwidth is required when --features is 'encodec'")

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

    start_time = time.time()
    vocos = load_vocos(args.features, device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    test_vocos(vocos, test_set, args.features, args.bandwidth, device)


if __name__ == '__main__':
    main()
