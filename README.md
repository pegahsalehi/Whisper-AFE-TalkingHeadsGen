# Comparative Analysis of Audio Feature Extraction for Real-Time Talking Portrait Synthesis

## Result based on RAD-NeRF

[![Watch the first video](https://img.youtube.com/vi/sBZWHk8y8-U/0.jpg)](https://youtu.be/sBZWHk8y8-U)

## Result based on ER-NeRF

[![Watch the second video](https://img.youtube.com/vi/BqKS1KAfrhA/0.jpg)](https://youtu.be/BqKS1KAfrhA)


## Audio Pre-process

In our paper, we use DeepSpeech features for evaluation.

You should specify the type of audio feature by `--asr_model <deepspeech, esperanto, hubert>` when training and testing.

### DeepSpeech

To extract features with DeepSpeech, use the following command:

```bash
python data_utils/deepspeech_features/extract_ds_features.py --input data/<name>.wav # save to data/

### Wav2Vec
You can also try to extract audio features via Wav2Vec, as used in RAD-NeRF:

```bash
python data_utils/wav2vec.py --wav data/<name>.wav --save_feats # save to data/<name>_eo.npy

### HuBERT
In our test, the HuBERT extractor performs better for more languages, which has already been used in GeneFace.
```bash
# Borrowed from GeneFace. English pre-trained.
python data_utils/hubert.py --wav data/<name>.wav # save to data/<name>_hu.npy

This README template includes command examples and references to relevant projects. Replace `<name>` with the appropriate file name when using these commands. You can customize the links if required.

### whisper














