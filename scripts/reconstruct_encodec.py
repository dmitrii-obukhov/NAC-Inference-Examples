import argparse
import os
import time
import torch
import torchaudio
from encodec import EncodecModel


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


def load_encodec_model(model_type, bandwidth, device):
    if model_type == '24khz':
        model = EncodecModel.encodec_model_24khz()
    elif model_type == '48khz':
        model = EncodecModel.encodec_model_48khz()

    model.set_target_bandwidth(bandwidth)
    model.eval().to(device)
    return model


def run_encodec(model, audio):
    with torch.no_grad():
        # Encode the audio
        start_time = time.time()
        encoded_frames = model.encode(audio)
        processing_time = time.time() - start_time
        print(f"Encode Time: {processing_time:.4f} seconds")
        
        # Concatenate the encoded frames
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

        # Decode the audio
        start_time = time.time()
        reconstructed_audio = model.decode(encoded_frames)
        processing_time = time.time() - start_time
        print(f"Decode Time: {processing_time:.4f} seconds")
    
    return reconstructed_audio


def test_encodec(model, test_set, device):
    total_processing_time = 0
    total_input_duration = 0

    for audio_file, output_file in test_set:
        # Load the input audio and measure its duration
        audio, sr = load_audio(audio_file, target_sr=model.sample_rate, target_channels=model.channels, device=device)
        input_duration = audio.size(1) / sr
        total_input_duration += input_duration

        # Measure processing time
        start_time = time.time()
        reconstructed_audio = run_encodec(model, audio.unsqueeze(0)).squeeze(0)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # Calculate RTF
        rtf = processing_time / input_duration

        # Save the output audio
        torchaudio.save(output_file, reconstructed_audio.cpu(), sr)

        # Print RTF for this audio file
        print(f"Audio file: {audio_file}")
        print(f"Processing Time: {processing_time:.4f} seconds")
        print(f"Input Duration: {input_duration:.4f} seconds")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Calculate average RTF
    average_rtf = total_processing_time / total_input_duration
    print(f"\nAverage Real-Time Factor (RTF): {average_rtf:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test EnCodec model for audio reconstruction.")
    parser.add_argument("--model_type", type=str, required=True, choices=["24khz", "48khz"],
                        help="Specify model type for EnCodec: '24khz' or '48khz'")
    parser.add_argument("--bandwidth", type=float, required=True, choices=[1.5, 3.0, 6.0, 12.0, 24.0],
                        help="Specify bandwidth for EnCodec model (options: 1.5, 3.0, 6.0, 12.0, or 24.0 kbps). Note: 24 kbps is only available for the 48khz model.")
    parser.add_argument("--input_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, help="Path to save the output audio file")
    parser.add_argument("--input_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed audio files")

    args = args = parser.parse_args()
    if args.bandwidth == 1.5 and args.model_type == '48khz':
        parser.error("Bandwidth of 1.5 kbps is only supported for the 24khz model.")
    if args.bandwidth == 24.0 and args.model_type == '24khz':
        parser.error("Bandwidth of 24 kbps is only supported for the 48khz model.")

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
    model = load_encodec_model(args.model_type, args.bandwidth, device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    # Test model
    test_encodec(model, test_set, device)

if __name__ == '__main__':
    main()