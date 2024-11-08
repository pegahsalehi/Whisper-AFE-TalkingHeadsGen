import whisper
import torch
import soundfile as sf
import torch.nn.functional as F
import numpy as np
import librosa

C = 384  # Number of audio dimensions for Whisper
window_size = 16
padding = 8 # or 7


def get_feature_whisper(audio_file_path):

    # Load the Whisper model
    model = whisper.load_model("tiny")

    # Load an audio file
    audio_input, sample_rate = sf.read(audio_file_path)
    # Ensure the audio data is in the correct format and dtype
    audio_input = torch.tensor(audio_input, dtype=torch.float32)
    if len(audio_input.shape) > 1:
        audio_input = torch.mean(audio_input, dim=1, keepdim=True)
        audio_input = audio_input.squeeze()

    # Convert the audio tensor to a NumPy array for resampling
    audio_input_np = audio_input.cpu().numpy()

    # Check if the sample rate needs to be changed
    if sample_rate != 16000:
        audio_input_np = librosa.resample(audio_input_np, orig_sr=sample_rate, target_sr=16000)

    # Convert the resampled NumPy array back to a PyTorch tensor
    audio_input = torch.tensor(audio_input_np)

    # Pad or trim audio to the correct length for the model
    audio = whisper.pad_or_trim(audio_input)

    # Make log-Mel spectrogram and ensure correct dtype
    mel = whisper.log_mel_spectrogram(audio)

    # Add batch dimension to `mel`
    mel = mel.unsqueeze(0)  # Shape becomes (1, channels, length)

    # Move to the model's device and ensure float32 dtype
    mel = mel.to(model.device, dtype=torch.float32)

    # Forward pass to extract features
    with torch.no_grad():
        # Extract features using the encoder
        encoder_output = model.encoder(mel)

    # `encoder_output` now contains the extracted audio features
    # Shape is (batch_size, T, C) for the encoder output
    features = encoder_output.squeeze(0)  # Remove batch dimension for (T, C) shape

    # Ensure features are on the same device
    features = features.to(model.device)

    features_cpu = features.cpu()  # Move to CPU
    features = features_cpu.numpy()  # Convert to NumPy array

    # Convert to PyTorch tensor and set device
    features = torch.tensor(features, dtype=torch.float32).to(model.device)
    #print("shape of features:", features.shape)

    # Reshape features for unfolding operation
    features = features.view(-1, C).permute(1, 0).contiguous()  # Shape becomes (C, T)
    features = features.view(1, C, -1, 1)  # Shape becomes (1, C, T, 1)
    #print("shape of features:", features.shape)
    # Unfold the features using a sliding window
    unfolded_features = F.unfold(features, kernel_size=(window_size, 1), padding=(padding, 0), stride=(2, 1))

    # Reshape unfolded features to desired shape
    unfolded_features = unfolded_features.view(C, window_size, -1).permute(2, 1, 0).contiguous()

    # Print the initial shape of the features
    #print("shape of features:", unfolded_features.shape)

    return unfolded_features


import numpy as np
from pydub import AudioSegment
import tempfile
import torch

# Load audio file
audio = AudioSegment.from_wav("./data/speech.wav")

# Length of audio in milliseconds
length_audio = len(audio)

# Start and end times (in milliseconds)
start = 0
chunk_duration = 30000  # 30 seconds
end = chunk_duration

# List to store features
all_features = []

# Split audio and process chunks
while start < length_audio:

    # Create chunk
    chunk = audio[start:end]

    # Convert stereo audio to mono
    chunk = chunk.set_channels(1)

    if end < length_audio:
        # Export chunk to a temporary file and get features
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
            chunk.export(temp_audio_file.name, format="wav")

            # Run the get_feature function for each chunk
            features = get_feature_whisper(temp_audio_file.name)

            # Ensure the features tensor is moved to CPU before converting to NumPy
            if isinstance(features, torch.Tensor):
              features = features.cpu().detach().numpy()


            all_features.append(features)


        # Check if it's the last chunk
    if end >= length_audio:
        # Run specific function for the last chunk
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
            chunk.export(temp_audio_file.name, format="wav")
            features_numpy = get_feature_whisper(temp_audio_file.name)

            # Ensure the features tensor is moved to CPU before converting to NumPy
            if isinstance(features_numpy, torch.Tensor):
                features_numpy = features_numpy.cpu().detach().numpy()

            audio = AudioSegment.from_file(temp_audio_file.name)
            length_aud = len(audio)

            x = length_aud / 1000

            rounded_number = round(round(x, 2) * 25)

            features = features_numpy[:rounded_number, :, :]

            all_features.append(features)  # Save the last chunk's features


    # Update start and end times for the next chunk
    start += chunk_duration
    end += chunk_duration

    # Ensure the end time does not exceed the audio length
    if end > length_audio:
        end = length_audio

# Concatenate all feature arrays along the first axis
all_features_array = np.concatenate(all_features, axis=0)

# Save the concatenated features as a .npy file
np.save("./data/speech_whisper.npy", all_features_array)

print(all_features_array.shape)

