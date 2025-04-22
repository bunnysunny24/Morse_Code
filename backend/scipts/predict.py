import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import os
import argparse
import soundfile as sf
from pydub import AudioSegment

# Constants (must match the ones used for training)
SAMPLE_RATE = 22050
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
    '-....-': '-',  '-.-.--': '!',  '.--.-': '@',  '---...': ':',
    '.-.-': '+',    '-...-': '=',   '.-...': '&'
}

def convert_to_wav(input_path, output_path=None):
    """Convert various audio formats to WAV format"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.wav"
    
    # Check file extension to determine format
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    # Handle different formats
    if ext in ['.mp3', '.mp4', '.m4a', '.aac', '.ogg', '.flac', '.wav']:
        try:
            # Use pydub for conversion - it relies on ffmpeg under the hood
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            print(f"Converted {input_path} to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            return None
    else:
        print(f"Unsupported file format: {ext}")
        return None

def load_and_preprocess_audio(file_path, max_time_steps=None):
    """Load and preprocess audio file to extract mel spectrogram features"""
    # Load audio file with librosa
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
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
    
    # Normalize - handle case where all values are the same
    min_val = np.min(mel_spec_db)
    max_val = np.max(mel_spec_db)
    
    if max_val > min_val:
        # Normal case - values differ
        mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
    else:
        # Edge case - all values are the same
        mel_spec_db = np.zeros_like(mel_spec_db)
    
    # Standardize shape
    if max_time_steps is not None:
        freq_bins, time_steps = mel_spec_db.shape
        padded_feat = np.zeros((freq_bins, max_time_steps))
        padded_feat[:, :min(time_steps, max_time_steps)] = mel_spec_db[:, :min(time_steps, max_time_steps)]
        mel_spec_db = padded_feat
    
    # Add channel dimension for Conv2D
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    
    return mel_spec_db

def window_audio(file_path, window_size=0.2, hop_size=0.05):
    """Break audio into smaller segments for analysis"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    windows = []
    times = []
    
    for i in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[i:i + window_samples]
        windows.append(window)
        times.append(i / sr)  # Time in seconds
    
    return windows, times, sr

def extract_morse_code(model, audio_file):
    """Extract Morse code elements from an audio file using windowing approach"""
    # Get model's time dimension size
    max_time_steps = model.input_shape[2]
    
    # Get audio windows
    windows, times, sr = window_audio(audio_file)
    print(f"Processing {len(windows)} windows from audio...")
    
    # Process each window
    predictions = []
    confidences = []
    
    # Create temp directory for window files
    temp_dir = "temp_windows"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Process every window and get prediction
        for i, window in enumerate(windows):
            # Save window to a temporary file
            temp_file = os.path.join(temp_dir, f"window_{i}.wav")
            sf.write(temp_file, window, sr)
            
            # Process the window
            features = load_and_preprocess_audio(temp_file, max_time_steps)
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            
            # Predict
            pred = model.predict(features, verbose=0)
            element_idx = np.argmax(pred[0])
            confidence = pred[0][element_idx]
            
            # Only keep predictions with good confidence
            if confidence > 0.7:
                predictions.append(MORSE_ELEMENTS[element_idx])
                confidences.append(confidence)
            
            # Clean up
            os.remove(temp_file)
        
        # Clean up temp directory
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error during processing: {e}")
        # Ensure cleanup
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            os.rmdir(temp_dir)
    
    # Simplify the sequence by removing duplicates
    simplified = []
    if predictions:
        current = predictions[0]
        count = 1
        avg_confidence = confidences[0]
        
        for i in range(1, len(predictions)):
            if predictions[i] == current:
                count += 1
                avg_confidence = (avg_confidence * (count - 1) + confidences[i]) / count
            else:
                if count >= 3:  # Only keep elements that appear at least 3 times in a row
                    simplified.append(current)
                current = predictions[i]
                count = 1
                avg_confidence = confidences[i]
        
        # Add the last element
        if count >= 3:
            simplified.append(current)
    
    return simplified

def morse_elements_to_text(elements):
    """Convert Morse elements to text"""
    # First, convert elements to Morse code
    morse_code = ""
    current_letter = ""
    text = ""
    
    # Map elements to Morse symbols
    morse_map = {'dot': '.', 'dash': '-'}
    
    for element in elements:
        if element in ['dot', 'dash']:
            current_letter += morse_map[element]
            morse_code += morse_map[element]
        elif element == 'short_pause' and current_letter:
            # End of a letter
            morse_code += " "
            if current_letter in MORSE_TO_TEXT:
                text += MORSE_TO_TEXT[current_letter]
            else:
                text += '?'  # Unknown Morse code
            current_letter = ""
        elif element == 'long_pause':
            # End of a word
            if current_letter:
                if current_letter in MORSE_TO_TEXT:
                    text += MORSE_TO_TEXT[current_letter]
                else:
                    text += '?'  # Unknown Morse code
                current_letter = ""
            morse_code += "   "
            text += ' '
    
    # Add the last letter if there is one
    if current_letter:
        if current_letter in MORSE_TO_TEXT:
            text += MORSE_TO_TEXT[current_letter]
        else:
            text += '?'  # Unknown Morse code
    
    return text, morse_code

def decode_morse_audio(model_path, audio_path):
    """Main function to decode Morse code from audio"""
    # Check if we need to convert the audio file
    _, ext = os.path.splitext(audio_path)
    if ext.lower() != '.wav':
        print(f"Converting {ext} file to WAV format...")
        converted_path = convert_to_wav(audio_path)
        if converted_path:
            audio_path = converted_path
        else:
            print("Conversion failed. Trying to use original file.")
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    print(f"Processing audio file: {audio_path}")
    morse_elements = extract_morse_code(model, audio_path)
    
    print(f"Detected {len(morse_elements)} Morse elements:")
    print(morse_elements)
    
    text, morse_code = morse_elements_to_text(morse_elements)
    
    print(f"\nMorse Code: {morse_code}")
    print(f"Decoded Text: {text}")
    
    # Clean up converted file if it was created
    if audio_path.endswith("_converted.wav") and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"Temporary converted file removed: {audio_path}")
        except:
            pass
    
    return text, morse_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morse Code Audio Decoder")
    parser.add_argument("--audio", required=True, help="Path to the Morse code audio file")
    parser.add_argument("--model", default="D:\\Bunny\\MorseCode\\backend\\scipts\\saved_models\\morse_recognition_model_v3.h5",
                       help="Path to the trained model")
    
    args = parser.parse_args()
    
    decode_morse_audio(args.model, args.audio)