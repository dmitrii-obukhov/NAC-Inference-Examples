#!/bin/bash


# EnCodec Examples

# Reconstruct using EnCodec with 24khz model and bandwidth 1.5 for a single file
python scripts/reconstruct_encodec.py --model_type 24khz --bandwidth 1.5 --input_file data/260_123288_000006_000002.wav --output_file output/Encodec_24khz_1.5/260_123288_000006_000002.wav

# Reconstruct using EnCodec with 24khz model and bandwidth 6.0 for a single file
python scripts/reconstruct_encodec.py --model_type 24khz --bandwidth 6.0 --input_file data/260_123288_000006_000002.wav --output_file output/Encodec_24khz_6.0/260_123288_000006_000002.wav

# Reconstruct using EnCodec with 48khz model and bandwidth 24.0 for a single file
python scripts/reconstruct_encodec.py --model_type 48khz --bandwidth 24.0 --input_file data/260_123288_000006_000002.wav --output_file output/Encodec_48khz_24.0/260_123288_000006_000002.wav

# Reconstruct using EnCodec with 24khz model and bandwidth 6.0 for all files in a directory
python scripts/reconstruct_encodec.py --model_type 24khz --bandwidth 6.0 --input_dir data --output_dir output/Encodec_24khz_6.0

# Reconstruct using EnCodec with 48khz model and bandwidth 12.0 for all files in a directory
python scripts/reconstruct_encodec.py --model_type 48khz --bandwidth 12.0 --input_dir data --output_dir output/Encodec_48khz_12.0


# FACodec Examples

# Reconstruct using FACodec from lifeiteng for a single file
python scripts/reconstruct_facodec_lifeiteng.py --input_file data/260_123288_000006_000002.wav --output_file output/FACodec/260_123288_000006_000002.wav

# Reconstruct using FACodec from lifeiteng for all files in a directory
python scripts/reconstruct_facodec_lifeiteng.py --input_dir data --output_dir output/FACodec

# Reconstruct using FAcodec from Platchaa for a single file
python scripts/reconstruct_facodec_platchaa.py --input_file data/260_123288_000006_000002.wav --output_file output/FAcodec_platchaa/260_123288_000006_000002.wav

# Reconstruct using FAcodec from Platchaa for all files in a directory
python scripts/reconstruct_facodec_platchaa.py --input_dir data --output_dir output/FAcodec_platchaa


# Language-Codec Examples

# Reconstruct using Language-Codec for a single file
python scripts/reconstruct_languagecodec.py --input_file data/260_123288_000006_000002.wav --output_file output/Language-Codec/260_123288_000006_000002.wav

# Reconstruct using Language-Codec for all files in a directory
python scripts/reconstruct_languagecodec.py --input_dir data --output_dir output/Language-Codec


# SpeechTokenizer Examples

# Reconstruct using SpeechTokenizer with hubert_avg model for a single file
python scripts/reconstruct_speech_tokenizer.py --model_id hubert_avg --input_file data/260_123288_000006_000002.wav --output_file output/SpeechTokenizer_hubert_avg/260_123288_000006_000002.wav

# Reconstruct using SpeechTokenizer with snake model for a single file
python scripts/reconstruct_speech_tokenizer.py --model_id snake --input_file data/260_123288_000006_000002.wav --output_file output/SpeechTokenizer_snake/260_123288_000006_000002.wav

# Reconstruct using SpeechTokenizer with hubert_avg model for all files in a directory
python scripts/reconstruct_speech_tokenizer.py --model_id hubert_avg --input_dir data --output_dir output/SpeechTokenizer_hubert_avg

# Reconstruct using SpeechTokenizer with snake model for all files in a directory
python scripts/reconstruct_speech_tokenizer.py --model_id snake --input_dir data --output_dir output/SpeechTokenizer_snake


# Vocos Examples

# Reconstruct using Vocos with mel features for a single file
python scripts/reconstruct_vocos.py --features mel --input_file data/260_123288_000006_000002.wav --output_file output/Vocos_mel/260_123288_000006_000002.wav

# Reconstruct using Vocos with encodec features and bandwidth 12.0 for a single file
python scripts/reconstruct_vocos.py --features encodec --bandwidth 12.0 --input_file data/260_123288_000006_000002.wav --output_file output/Vocos_encodec_12.0/260_123288_000006_000002.wav

# Reconstruct using Vocos with mel features for all files in a directory
python scripts/reconstruct_vocos.py --features mel --input_dir data --output_dir output/Vocos_mel

# Reconstruct using Vocos with encodec features and bandwidth 1.5 for all files in a directory
python scripts/reconstruct_vocos.py --features encodec --bandwidth 1.5 --input_dir data --output_dir output/Vocos_encodec_1.5

# Reconstruct using Vocos with encodec features and bandwidth 3.0 for all files in a directory
python scripts/reconstruct_vocos.py --features encodec --bandwidth 3.0 --input_dir data --output_dir output/Vocos_encodec_3.0

# Reconstruct using Vocos with encodec features and bandwidth 6.0 for all files in a directory
python scripts/reconstruct_vocos.py --features encodec --bandwidth 6.0 --input_dir data --output_dir output/Vocos_encodec_6.0

# Reconstruct using Vocos with encodec features and bandwidth 12.0 for all files in a directory
python scripts/reconstruct_vocos.py --features encodec --bandwidth 12.0 --input_dir data --output_dir output/Vocos_encodec_12.0


# WavTokenizer Examples

# Reconstruct using WavTokenizer with WavTokenizer_small_600_24k_4096 for a single file
python scripts/reconstruct_wavtokenizer.py --model_name WavTokenizer_small_600_24k_4096 --input_file data/260_123288_000006_000002.wav --output_file output/WavTokenizer_small_600_24k_4096/260_123288_000006_000002.wav

# Reconstruct using WavTokenizer with WavTokenizer_small_320_24k_4096 for a single file
python scripts/reconstruct_wavtokenizer.py --model_name WavTokenizer_small_320_24k_4096 --input_file data/260_123288_000006_000002.wav --output_file output/WavTokenizer_small_320_24k_4096/260_123288_000006_000002.wav

# Reconstruct using WavTokenizer with wavtokenizer_medium_speech_320_24k_v2 for a single file
python scripts/reconstruct_wavtokenizer.py --model_name wavtokenizer_medium_speech_320_24k_v2 --input_file data/260_123288_000006_000002.wav --output_file output/wavtokenizer_medium_speech_320_24k_v2/260_123288_000006_000002.wav

# Reconstruct using WavTokenizer with WavTokenizer_small_600_24k_4096 for all files in a directory
python scripts/reconstruct_wavtokenizer.py --model_name WavTokenizer_small_600_24k_4096 --input_dir data --output_dir output/WavTokenizer_small_600_24k_4096

# Reconstruct using WavTokenizer with WavTokenizer_small_320_24k_4096 for all files in a directory
python scripts/reconstruct_wavtokenizer.py --model_name WavTokenizer_small_320_24k_4096 --input_dir data --output_dir output/WavTokenizer_small_320_24k_4096

# Reconstruct using WavTokenizer with wavtokenizer_medium_speech_320_24k_v2 for all files in a directory
python scripts/reconstruct_wavtokenizer.py --model_name wavtokenizer_medium_speech_320_24k_v2 --input_dir data --output_dir output/wavtokenizer_medium_speech_320_24k_v2
