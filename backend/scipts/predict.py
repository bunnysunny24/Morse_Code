import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import os
import argparse
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile

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

def load_and_preprocess_audio(file_path, max_time_steps=None):
    """
    Load and preprocess audio file to extract mel spectrogram features
    
    Args:
        file_path: Path to the audio file
        max_time_steps: Maximum time steps for padding
        
    Returns:
        Mel spectrogram features
    """
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
    
    # Normalize
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
    
    # Standardize shape if max_time_steps is provided
    if max_time_steps is not None:
        freq_bins, time_steps = mel_spec_db.shape
        padded_feat = np.zeros((freq_bins, max_time_steps))
        padded_feat[:, :min(time_steps, max_time_steps)] = mel_spec_db[:, :min(time_steps, max_time_steps)]
        mel_spec_db = padded_feat
    
    return mel_spec_db

def window_audio_file(file_path, window_size=0.2, hop_length=0.05):
    """
    Split audio into windows for more detailed analysis
    
    Args:
        file_path: Path to audio file
        window_size: Size of each window in seconds
        hop_length: Hop length between windows in seconds
        
    Returns:
        List of windowed audio samples, sample rate
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Convert window size and hop length from seconds to samples
    window_samples = int(window_size * sr)
    hop_samples = int(hop_length * sr)
    
    # Create windows
    windows = []
    for i in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[i:i + window_samples]
        windows.append(window)
    
    return windows, sr

def predict_element(model, audio_features):
    """
    Predict a single Morse code element from preprocessed audio features
    """
    # Add batch dimension
    features_batch = np.expand_dims(audio_features, axis=0)
    
    # Make prediction
    predictions = model.predict(features_batch, verbose=0)
    
    # Get highest probability class
    predicted_idx = np.argmax(predictions[0])
    predicted_element = MORSE_ELEMENTS[predicted_idx]
    
    # Also get confidence score
    confidence = predictions[0][predicted_idx]
    
    return predicted_element, confidence

def predict_from_whole_file(model, audio_file, max_time_steps):
    """
    Predict a Morse code element from an entire audio file
    """
    # Load and preprocess audio
    features = load_and_preprocess_audio(audio_file, max_time_steps)
    
    # Add channel dimension for Conv2D
    features = np.expand_dims(features, axis=-1)
    
    # Get prediction
    element, confidence = predict_element(model, features)
    
    return element, confidence

def predict_from_windows(model, audio_file, max_time_steps, window_size=0.2, hop_length=0.05):
    """
    Predict Morse code elements using a sliding window approach
    """
    # Get audio windows
    windows, sr = window_audio_file(audio_file, window_size, hop_length)
    
    # Process each window
    predictions = []
    confidences = []
    
    print(f"Processing {len(windows)} windows...")
    
    # Create a temporary directory for window files
    tmp_dir = os.path.join(os.path.dirname(audio_file), "temp_windows")
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        for i, window in enumerate(windows):
            # Save window to temp file
            tmp_file = os.path.join(tmp_dir, f"window_{i}.wav")
            wavfile.write(tmp_file, sr, window)
            
            # Preprocess window
            features = load_and_preprocess_audio(tmp_file, max_time_steps)
            features = np.expand_dims(features, axis=-1)  # Add channel dimension
            
            # Get prediction
            element, confidence = predict_element(model, features)
            predictions.append(element)
            confidences.append(confidence)
            
            # Remove temp file
            os.remove(tmp_file)
            
        # Clean up
        os.rmdir(tmp_dir)
    except Exception as e:
        print(f"Error during window processing: {e}")
        # Ensure cleanup even if error occurs
        if os.path.exists(tmp_dir):
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)
    
    return predictions, confidences

def compress_predictions(predictions, confidences, threshold=0.8, min_streak=3):
    """
    Compress predictions by removing noise and consolidating streaks
    """
    if not predictions:
        return []
    
    compressed = []
    current = predictions[0]
    count = 1
    avg_confidence = confidences[0]
    
    for i in range(1, len(predictions)):
        if predictions[i] == current:
            # Same element, continue streak
            count += 1
            avg_confidence = (avg_confidence * (count - 1) + confidences[i]) / count
        else:
            # Different element, check if current streak is valid
            if count >= min_streak and avg_confidence >= threshold:
                compressed.append(current)
            
            # Start new streak
            current = predictions[i]
            count = 1
            avg_confidence = confidences[i]
    
    # Don't forget the last streak
    if count >= min_streak and avg_confidence >= threshold:
        compressed.append(current)
    
    return compressed

def morse_elements_to_code(elements):
    """Convert Morse elements to dots and dashes"""
    morse_code = ""
    for element in elements:
        if element == "dot":
            morse_code += "."
        elif element == "dash":
            morse_code += "-"
        elif element == "short_pause":
            morse_code += " "  # Space between letters
        elif element == "long_pause":
            morse_code += "   "  # Space between words (3 spaces)
    return morse_code

def morse_to_text(morse_elements):
    """
    Convert a sequence of Morse code elements to text
    
    Args:
        morse_elements: List of Morse code elements ('dot', 'dash', 'short_pause', 'long_pause')
        
    Returns:
        Decoded text
    """
    # First convert to morse code string
    morse_code = ""
    current_letter = ""
    text = ""
    
    # Map elements to morse symbols
    morse_map = {'dot': '.', 'dash': '-'}
    
    for element in morse_elements:
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

def plot_predictions(predictions, confidences, output_file=None):
    """Plot predictions and their confidences over time"""
    # Map elements to numeric values for plotting
    element_map = {e: i for i, e in enumerate(MORSE_ELEMENTS)}
    numeric_preds = [element_map[p] for p in predictions]
    
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.subplot(2, 1, 1)
    plt.plot(numeric_preds, marker='o', linestyle='-', markersize=3)
    plt.yticks(range(len(MORSE_ELEMENTS)), MORSE_ELEMENTS)
    plt.title('Predicted Morse Elements Over Time')
    plt.xlabel('Window Index')
    plt.ylabel('Element')
    plt.grid(True)
    
    # Plot confidence
    plt.subplot(2, 1, 2)
    plt.plot(confidences, marker='.', linestyle='-')
    plt.title('Prediction Confidence Over Time')
    plt.xlabel('Window Index')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Morse code audio decoder")
    parser.add_argument("--model", type=str, default="D:\\Bunny\\MorseCode\\backend\\scipts\\saved_models\\morse_recognition_model_v3.h5",
                        help="Path to trained model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV format)")
    parser.add_argument("--window", action="store_true", help="Use sliding window approach (recommended)")
    parser.add_argument("--plot", action="store_true", help="Plot predictions over time")
    parser.add_argument("--output", type=str, help="Path to save output text")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} not found.")
        return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Get model input shape to determine max_time_steps
    max_time_steps = model.input_shape[2]  # Should be the time dimension
    print(f"Model expects input with time steps: {max_time_steps}")
    
    start_time = time.time()
    
    # Process audio
    if args.window:
        print("Using sliding window approach...")
        predictions, confidences = predict_from_windows(model, args.audio, max_time_steps)
        
        # Print window predictions if requested
        print(f"Raw predictions from {len(predictions)} windows:")
        print(predictions[:20] + ['...'] if len(predictions) > 20 else predictions)
        
        # Compress predictions
        compressed = compress_predictions(predictions, confidences)
        
        print(f"Compressed Morse elements ({len(compressed)}): {compressed}")
        
        if args.plot:
            plot_file = args.output.replace('.txt', '_plot.png') if args.output else None
            plot_predictions(predictions, confidences, plot_file)
        
    else:
        # Simple prediction for the entire file
        predicted_element, confidence = predict_from_whole_file(model, args.audio, max_time_steps)
        print(f"Predicted Morse element: {predicted_element} (confidence: {confidence:.2f})")
        compressed = [predicted_element]
    
    # Convert to text
    decoded_text, morse_code = morse_to_text(compressed)
    
    # Print results
    print("\nResults:")
    print(f"Morse Code: {morse_code}")
    print(f"Decoded Text: {decoded_text}")
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Morse Code: {morse_code}\n")
            f.write(f"Decoded Text: {decoded_text}\n")
            
            # Additional details if window approach was used
            if args.window:
                f.write(f"\nDetailed Analysis:\n")
                f.write(f"Number of Windows: {len(predictions)}\n")
                f.write(f"Compressed Elements: {compressed}\n")
        
        print(f"Results saved to {args.output}")
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    print(f"Morse Code Predictor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()