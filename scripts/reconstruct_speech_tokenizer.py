import argparse
import os
import time
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from speechtokenizer import SpeechTokenizer


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


def load_speech_tokenizer(model_id, device):
    if model_id == "hubert_avg":
        config_path = hf_hub_download(repo_id="fnlp/SpeechTokenizer", filename="speechtokenizer_hubert_avg/config.json")
        ckpt_path = hf_hub_download(repo_id="fnlp/SpeechTokenizer", filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    elif model_id == "snake":
        config_path = hf_hub_download(repo_id="fnlp/AnyGPT-speech-modules", filename="speechtokenizer/config.json")
        ckpt_path = hf_hub_download(repo_id="fnlp/AnyGPT-speech-modules", filename="speechtokenizer/ckpt.dev")
    
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval()
    model.to(device)
    
    return model


def run_speech_tokenizer(model, audio):
    # Extract discrete codes from SpeechTokenizer
    start_time = time.time()
    with torch.no_grad():
        codes = model.encode(audio)  # codes: (n_q, B, T)
    processing_time = time.time() - start_time
    print(f"Encode Time: {processing_time:.4f} seconds")

    RVQ_1 = codes[:1, :, :]  # Contain content info, can be considered as semantic tokens
    RVQ_supplement = codes[1:, :, :]  # Contain timbre info, complete info lost by the first quantizer

    # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
    start_time = time.time()
    audio_reconstructed = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))
    processing_time = time.time() - start_time
    print(f"Decode Time: {processing_time:.4f} seconds")
    
    return audio_reconstructed


def test_speech_tokenizer(model, test_set, device):
    total_processing_time = 0
    total_input_duration = 0

    for audio_file, output_file in test_set:
        # Load the input audio and measure its duration
        audio, sr = load_audio(audio_file, model.sample_rate, device)
        input_duration = audio.size(1) / sr
        total_input_duration += input_duration

        # Measure processing time
        start_time = time.time()
        audio_reconstructed = run_speech_tokenizer(model, audio.unsqueeze(0)).squeeze(0)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # Calculate RTF
        rtf = processing_time / input_duration

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
    parser = argparse.ArgumentParser(description="Test SpeechTokenizer model with specified configurations.")
    parser.add_argument("--model_id", type=str, required=True, choices=["hubert_avg", "snake"],
                        help="Choose the model for SpeechTokenizer: 'hubert_avg' or 'snake'")
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

    start_time = time.time()
    model = load_speech_tokenizer(args.model_id, device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    test_speech_tokenizer(model, test_set, device)


if __name__ == '__main__':
    main()
