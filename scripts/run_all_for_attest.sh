#!/bin/bash

input_dir="data"

# Function to create filelist.txt
create_filelist() {
  local filelist_path="$1"
  mkdir -p $(dirname "$filelist_path")
  > "$filelist_path"
  for file in "$input_dir"/*.wav; do
    filename=$(basename "$file")
    echo "${filename}|" >> "$filelist_path"
  done
}

# Create output directory for attest format
projects=(
  "Encodec_24khz_1.5" "Encodec_24khz_3.0" "Encodec_24khz_6.0" "Encodec_24khz_12.0" 
  "Encodec_48khz_3.0" "Encodec_48khz_6.0" "Encodec_48khz_12.0" "Encodec_48khz_24.0" 
  "FACodec_lifeiteng" "FAcodec_platchaa"
  "LanguageCodec"
  "SpeechTokenizer_hubert_avg" "SpeechTokenizer_snake"
  "Vocos_mel" "Vocos_encodec_1.5" "Vocos_encodec_3.0" "Vocos_encodec_6.0" "Vocos_encodec_12.0"
  "WavTokenizer_small_600_24k_4096" "WavTokenizer_small_320_24k_4096" "wavtokenizer_medium_speech_320_24k_v2"
)

for project in "${projects[@]}"; do
  echo "========================================"
  echo "Infering ${project}..."
  mkdir -p output_for_attest/${project}/wavs

  # Reconstruct using Vocos, SpeechTokenizer, FACodec, EnCodec, Language-Codec, or WavTokenizer for all files in a directory
  if [[ "$project" == Encodec_24khz* ]]; then
    bandwidth=$(echo $project | cut -d'_' -f3)
    python scripts/reconstruct_encodec.py --model_type 24khz --bandwidth $bandwidth --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == Encodec_48khz* ]]; then
    bandwidth=$(echo $project | cut -d'_' -f3)
    python scripts/reconstruct_encodec.py --model_type 48khz --bandwidth $bandwidth --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == FACodec_lifeiteng ]]; then
    python scripts/reconstruct_facodec_lifeiteng.py --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == FAcodec_platchaa ]]; then
    python scripts/reconstruct_facodec_platchaa.py --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == LanguageCodec ]]; then
    python scripts/reconstruct_languagecodec.py --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == SpeechTokenizer_hubert_avg ]]; then
    python scripts/reconstruct_speech_tokenizer.py --model_id hubert_avg --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == SpeechTokenizer_snake ]]; then
    python scripts/reconstruct_speech_tokenizer.py --model_id snake --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == Vocos_mel ]]; then
    python scripts/reconstruct_vocos.py --features mel --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == Vocos_encodec* ]]; then
    bandwidth=$(echo $project | cut -d'_' -f3)
    python scripts/reconstruct_vocos.py --features encodec --bandwidth $bandwidth --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == WavTokenizer_small_600_24k_4096 ]]; then
    python scripts/reconstruct_wavtokenizer.py --model_name WavTokenizer_small_600_24k_4096 --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == WavTokenizer_small_320_24k_4096 ]]; then
    python scripts/reconstruct_wavtokenizer.py --model_name WavTokenizer_small_320_24k_4096 --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  elif [[ "$project" == wavtokenizer_medium_speech_320_24k_v2 ]]; then
    python scripts/reconstruct_wavtokenizer.py --model_name wavtokenizer_medium_speech_320_24k_v2 --input_dir "$input_dir" --output_dir output_for_attest/${project}/wavs
  fi

  # Create the filelist.txt for each project
  filelist_path="output_for_attest/${project}/meta/filelist.txt"
  create_filelist "$filelist_path"

done

# Create reference project
mkdir -p output_for_attest/Reference/wavs
cp "$input_dir"/*.wav output_for_attest/Reference/wavs/
filelist_path="output_for_attest/Reference/meta/filelist.txt"
create_filelist "$filelist_path"
