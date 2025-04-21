import numpy as np
import tensorflow as tf
import librosa
import argparse
import os

# Constants - must match those used for training
SAMPLE_RATE = 22050
DURATION = 30  # max duration in seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MORSE_ELEMENTS = ['dot', 'dash', 'short_pause', 'long_pause']

# Morse code dictionary for decoding
MORSE_TO_TEXT = {
    '.-': 'A',     '-...': 'B',   '-.-.': 'C',   '-..': 'D',
    '.': 'E',      '..-.': 'F',   '--.': 'G',    '....': 'H',
    '..': 'I',     '.---': 'J',   '-.-': 'K',    '.-..': 'L',
    '--': 'M',     '-.': 'N',     '---': 'O',    '.--.': 'P',
    '--.-': 'Q',   '.-.': 'R',    '...': 'S',    '-': 'T',
    '..-': 'U',    '...-': 'V',   '.--': 'W',    '-..-': 'X',
    '-.--': 'Y',   '--..': 'Z',
    '.----': '1',  '..---': '2',  '...--': '3',  '....-': '4',
    '.....': '5',  '-....': '6',  '--...': '7',  '---..': '8',
    '----.': '9',  '-----': '0',
    '.-.-.-': '.',  '--..--': ',',  '..--..': '?',  '-..-.': '/',
    '-....-': '-',  '-.-.--': '!'
}

def load_and_preprocess_audio(file_path):
    """
    Load and preprocess audio file to extract mel spectrogram features
    """
    # Load audio file with librosa
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Extract mel spectrogram features
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
    
    return mel_spec_db

def decode_morse_sequence(sequence):
    """
    Decode a sequence of Morse elements into text
    """
    current_char = []
    message = []
    
    for element in sequence:
        if element == 'dot':
            current_char.append('.')
        elif element == 'dash':
            current_char.append('-')
        elif element == 'short_pause' and current_char:
            # End of character
            morse_char = ''.join(current_char)
            if morse_char in MORSE_TO_TEXT:
                message.append(MORSE_TO_TEXT[morse_char])
            current_char = []
        elif element == 'long_pause':
            # End of word
            if current_char:
                morse_char = ''.join(current_char)
                if morse_char in MORSE_TO_TEXT:
                    message.append(MORSE_TO_TEXT[morse_char])
                current_char = []
            message.append(' ')
    
    # Handle any remaining character
    if current_char:
        morse_char = ''.join(current_char)
        if morse_char in MORSE_TO_TEXT:
            message.append(MORSE_TO_TEXT[morse_char])
    
    return ''.join(message).strip()

def predict_from_audio(model, audio_file):
    """
    Predict Morse code elements from audio file and decode to text
    """
    # Load and preprocess audio
    features = load_and_preprocess_audio(audio_file)
    
    # Reshape for model input (add batch dimension)
    features = np.expand_dims(features, axis=0)
    
    # Make prediction
    predictions = model.predict(features)
    
    # Get highest probability class for each time step
    predicted_classes = np.argmax(predictions, axis=-1)
    
    # Convert class indices to Morse elements
    morse_sequence = [MORSE_ELEMENTS[idx] for idx in predicted_classes[0]]
    
    # Decode Morse sequence to text
    decoded_text = decode_morse_sequence(morse_sequence)
    
    # Also return the raw Morse code
    morse_code = ''.join(['.' if elem == 'dot' else 
                         '-' if elem == 'dash' else 
                         ' ' if elem == 'short_pause' else
                         ' / ' if elem == 'long_pause' else ''
                         for elem in morse_sequence])
    
    return decoded_text, morse_code

def main():
    parser = argparse.ArgumentParser(description='Decode Morse code from audio')
    parser.add_argument('audio_file', help='Path to Morse code audio file')
    parser.add_argument('--model', default='morse_recognition_model.h5', help='Path to trained model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file {args.audio_file} not found.")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    
    # Process the audio file
    print(f"Decoding Morse code from {args.audio_file}...")
    decoded_text, morse_code = predict_from_audio(model, args.audio_file)
    
    print("\nResults:")
    print(f"Decoded Morse code: {morse_code}")
    print(f"Decoded text: {decoded_text}")

if __name__ == "__main__":
    main()