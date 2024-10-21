# NAC-Inference-Examples

This project provides hands-on examples of how to run inference with various neural audio codecs (NACs). The focus is on codecs that have been used in speech synthesis. It includes scripts that generate output in an ATTEST-friendly format, allowing you to directly compute metrics with [ATTEST](https://github.com/constructor-tech/attest-speech-analysis-tool). Use this repository to explore and evaluate different NAC approaches.

## Installation

Python3.10 is recommended. Install the requirements using:

```
git clone https://github.com/dmitrii-obukhov/NAC-Inference-Examples.git

cd NAC-Inference-Examples

git submodule update --init --recursive

pip install -r requirements.txt
```

## Usage

This repository includes inference examples for the following neural audio codecs:

 - [EnCodec](https://github.com/facebookresearch/encodec)
 - [FACodec (lifeiteng)](https://github.com/lifeiteng/naturalspeech3_facodec)
 - [FACodec (Platchaa)](https://github.com/Plachtaa/FAcodec)
 - [Language-Codec](https://github.com/jishengpeng/Languagecodec)
 - [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
 - [Vocos](https://github.com/gemelo-ai/vocos)
 - [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)

### EnCodec

To reconstruct audio with EnCodec, use the following command:

```
python scripts/reconstruct_encodec.py --model_type [24khz or 48khz] --bandwidth [1.5, 3.0, 6.0, 12.0, or 24.0] --input_file [input file] --output_file [output file]
```

Optional Parameters:
 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` / `--output_file` to process all files in a directory.

### FACodec (lifeiteng)

To reconstruct audio with FACodec (lifeiteng), use the following command:

```
python scripts/reconstruct_facodec_lifeiteng.py --input_file [input file] --output_file [output file]
```

Optional Parameters:
 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` / `--output_file` to process all files in a directory.

### FACodec (Platchaa)

To reconstruct audio with FACodec (Platchaa), use the following command:

```
python scripts/reconstruct_facodec_platchaa.py --input_file [input file] --output_file [output file]
```

Optional Parameters:

 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` /` --output_file` to process all files in a directory.

### Language-Codec

To reconstruct audio with Language-Codec, use the following command:

```
python scripts/reconstruct_languagecodec.py --input_file [input file] --output_file [output file]
```

Optional Parameters:

 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` / `--output_file` to process all files in a directory.

### SpeechTokenizer

To reconstruct audio with SpeechTokenizer, use the following command:

```
python scripts/reconstruct_speech_tokenizer.py --model_id [hubert_avg or snake] --input_file [input file] --output_file [output file]
```

Optional Parameters:
 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` / `--output_file` to process all files in a directory.

### Vocos

To reconstruct audio with Vocos, use the following command:

```
python scripts/reconstruct_vocos.py --features [mel or encodec] --input_file [input file] --output_file [output file]
```

Optional Parameters:
 - `--bandwidth`: Specify when feature is set to encodec (options: 1.5, 3.0, 6.0, or 12.0). The default is 12.0.
 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` / `--output_file` to process all files in a directory.

### WavTokenizer

To reconstruct audio with WavTokenizer, use the following command:

```
python scripts/reconstruct_wavtokenizer.py --model_name [model_name] --input_file [input file] --output_file [output file]
```

Where `model_name` in on of `WavTokenizer_small_600_24k_4096`, `WavTokenizer_small_320_24k_4096` or `wavtokenizer_medium_speech_320_24k_v2`.

Optional Parameters:

 - `--input_dir` / `--output_dir`: Specify these options instead of `--input_file` / `--output_file` to process all files in a directory.

## Examples

This script demonstrates how to run inference for different models [scripts/run_examples.sh](scripts/run_examples.sh).

## ATTEST Integration

Use the following script to prepare data in an ATTEST-friendly format: [scripts/run_all_for_attest.sh](scripts/run_all_for_attest.sh).

This script reconstructs audio into `output_for_attest`, organizing it according to ATTEST’s required format. It also creates a reference project from the original input data to make comparison.

[ATTEST](https://github.com/constructor-tech/attest-speech-analysis-tool) will provide you with a report containing metrics. Metrics descriptions and additional metrics can be found in the [ATTEST documentation](https://github.com/constructor-tech/attest-speech-analysis-tool/blob/main/README.md). 

| Metric                                |   UTMOS ↑ |   SpeechBERTScore ↑ |   Squim STOI ↑ |   Squim PESQ ↑ |   Squim SI-SDR ↑ | Speaker Similarity (ECAPA-TDNN) ↑ | Reconstruction RTF |
|:--------------------------------------|----------:|--------------------:|---------------:|---------------:|-----------------:|----------------------------------:|-------------------:|
| Reference                             |   4.2505  |            1        |       0.990466 |        3.67205 |         25.0494  |                          1        |                    |
| Encodec_24khz_1.5                     |   1.69883 |            0.87291  |       0.873847 |        1.54732 |          4.65589 |                          0.51568  |             0.0254 |
| Encodec_24khz_3.0                     |   2.49831 |            0.935515 |       0.930555 |        2.20071 |          9.79701 |                          0.762106 |             0.0222 |
| Encodec_24khz_6.0                     |   3.37016 |            0.961427 |       0.972886 |        2.91298 |         16.9537  |                          0.86857  |             0.0225 |
| Encodec_24khz_12.0                    |   3.80225 |            0.974682 |       0.985861 |        3.37845 |         22.1408  |                          0.910487 |             0.0248 |
| Encodec_48khz_3.0                     |   1.68662 |            0.884645 |       0.89841  |        1.84527 |          7.97226 |                          0.588704 |             0.0581 |
| Encodec_48khz_6.0                     |   2.85702 |            0.945446 |       0.952297 |        2.47492 |         13.4713  |                          0.77532  |             0.0566 |
| Encodec_48khz_12.0                    |   3.5481  |            0.967346 |       0.976078 |        3.02159 |         18.8814  |                          0.848866 |             0.0563 |
| Encodec_48khz_24.0                    |   3.88944 |            0.977586 |       0.986879 |        3.39461 |         21.5628  |                          0.89282  |             0.0591 |
| FACodec_lifeiteng                     |   4.27953 |            0.967634 |       0.992411 |        3.77985 |         26.6268  |                          0.870559 |             0.0269 |
| FAcodec_platchaa                      |   3.7256  |            0.928103 |       0.989421 |        3.47873 |         23.6755  |                          0.667826 |             0.0746 |
| LanguageCodec                         |   4.11773 |            0.975021 |       0.992009 |        3.58266 |         24.8315  |                          0.943377 |             0.0237 |
| SpeechTokenizer_hubert_avg            |   4.09132 |            0.956058 |       0.992191 |        3.65802 |         25.0519  |                          0.854601 |             0.0349 |
| SpeechTokenizer_snake                 |   4.0697  |            0.953432 |       0.990394 |        3.63908 |         24.6742  |                          0.859049 |             0.0652 |
| Vocos_encodec_1.5                     |   3.41289 |            0.90202  |       0.990149 |        3.55445 |         21.0133  |                          0.708921 |             0.0215 |
| Vocos_encodec_3.0                     |   3.87598 |            0.950471 |       0.993601 |        3.68377 |         24.7166  |                          0.858097 |             0.0201 |
| Vocos_encodec_6.0                     |   3.97273 |            0.964828 |       0.994037 |        3.7618  |         24.9259  |                          0.914804 |             0.0202 |
| Vocos_encodec_12.0                    |   4.00707 |            0.974555 |       0.99208  |        3.71123 |         25.5214  |                          0.94663  |             0.0204 |
| Vocos_mel                             |   3.92117 |            0.980863 |       0.991091 |        3.5926  |         22.7136  |                          0.985403 |             0.0056 |
| WavTokenizer_small_320_24k_4096       |   4.17364 |            0.939801 |       0.995624 |        3.78539 |         25.7049  |                          0.82858  |             0.0253 |
| WavTokenizer_small_600_24k_4096       |   3.73223 |            0.909782 |       0.989835 |        3.56292 |         21.0606  |                          0.707959 |             0.0221 |
| wavtokenizer_medium_speech_320_24k_v2 |   4.13652 |            0.938756 |       0.993538 |        3.81021 |         25.8433  |                          0.768522 |             0.0251 |



Notes:
 - The table is generated using ATTEST, with the last column (RTF) added manually. The RTF was measured on an NVIDIA GeForce GTX 1660 Ti.
 - The provided metrics are just an example for the audio files in the data folder. They do not represent a comprehensive benchmark.
