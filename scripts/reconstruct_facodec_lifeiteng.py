import argparse
import os
import soundfile as sf
import time
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from ns3_codec import FACodecEncoder, FACodecDecoder


SAMPLE_RATE = 16000


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


def load_facodec(device):
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))

    fa_encoder.eval().to(device)
    fa_decoder.eval().to(device)

    return fa_encoder, fa_decoder


def run_facodec(fa_encoder, fa_decoder, audio):
    with torch.no_grad():
        # encode
        start_time = time.time()
        enc_out = fa_encoder(audio)
        processing_time = time.time() - start_time
        print(f"Encode Time: {processing_time:.4f} seconds")

        # quantize
        start_time = time.time()
        vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
        processing_time = time.time() - start_time
        print(f"Quantize Time: {processing_time:.4f} seconds")
        
        # decode
        start_time = time.time()
        recon_audio = fa_decoder.inference(vq_post_emb, spk_embs)
        processing_time = time.time() - start_time
        print(f"Decode Time: {processing_time:.4f} seconds")

        return recon_audio


def test_facodec(fa_encoder, fa_decoder, test_set, device):
    total_processing_time = 0
    total_input_duration = 0

    for audio_file, output_file in test_set:
        # Load the input audio and measure its duration
        audio, sr = load_audio(audio_file, target_sr=SAMPLE_RATE, device=device)
        input_duration = audio.size(1) / sr
        total_input_duration += input_duration

        # Measure processing time
        start_time = time.time()
        recon_audio = run_facodec(fa_encoder, fa_decoder, audio.unsqueeze(0)).squeeze(0)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        # Calculate RTF
        rtf = processing_time / input_duration

        # Save the output audio
        sf.write(output_file, recon_audio[0].cpu().numpy(), sr)

        # Print RTF for this audio file
        print(f"Audio file: {audio_file}")
        print(f"Processing Time: {processing_time:.4f} seconds")
        print(f"Input Duration: {input_duration:.4f} seconds")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Calculate average RTF
    average_rtf = total_processing_time / total_input_duration
    print(f"\nAverage Real-Time Factor (RTF): {average_rtf:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test FACodec model for audio reconstruction.")
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
    fa_encoder, fa_decoder = load_facodec(device)
    processing_time = time.time() - start_time
    print(f"Model Loading Time: {processing_time:.4f} seconds")

    # Test model
    test_facodec(fa_encoder, fa_decoder, test_set, device)


if __name__ == '__main__':
    main()