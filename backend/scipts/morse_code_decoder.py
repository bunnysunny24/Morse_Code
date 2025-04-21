import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Morse Code Dictionary (for decoding)
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0', ' ': ' '
}

# Reverse the dictionary for encoding
REVERSE_MORSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}

class MorseCodeDetector:
    def __init__(self, sample_rate=22050, frame_length=512, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.model = None
        self.label_map = {0: '.', 1: '-', 2: 'short_pause', 3: 'long_pause'}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def preprocess_audio(self, file_path, display=False):
        """Load and preprocess audio file to extract features"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Extract melspectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.frame_length,
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            
            # Convert to dB scale
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            if display:
                # Display the melspectrogram
                plt.figure(figsize=(12, 4))
                librosa.display.specshow(
                    mel_spectrogram_db, sr=sr, hop_length=self.hop_length,
                    x_axis='time', y_axis='mel'
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel Spectrogram')
                plt.tight_layout()
                plt.show()
                
                # Display the waveform
                plt.figure(figsize=(12, 4))
                librosa.display.waveshow(y, sr=sr)
                plt.title('Waveform')
                plt.tight_layout()
                plt.show()
                
            return mel_spectrogram_db
        except Exception as e:
            print(f"Error processing audio file {file_path}: {str(e)}")
            return None
    
    def segment_audio(self, audio, segment_length=128):
        """Segment the audio into fixed-sized chunks for model input"""
        # Transpose to have time as the first dimension
        audio = audio.T
        
        # Ensure we have complete segments
        n_segments = audio.shape[0] // segment_length
        if n_segments == 0:
            # If audio is shorter than segment_length, pad it
            padded = np.zeros((segment_length, audio.shape[1]))
            padded[:audio.shape[0], :] = audio
            segments = np.array([padded])
        else:
            # Otherwise, split into segments
            segments = np.array([audio[i*segment_length:(i+1)*segment_length] 
                              for i in range(n_segments)])
        
        return segments
    
    def build_model(self, input_shape):
        """Build a CNN model for Morse code detection"""
        model = Sequential([
            # First Conv layer
            Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            
            # Second Conv layer
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # Third Conv layer
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')  # 4 classes: dot, dash, short pause, long pause
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_dataset(self, data_dir):
        """Prepare dataset from directory of audio files with labels"""
        features = []
        labels = []
        
        # Expected directory structure:
        # data_dir/
        #   dots/
        #   dashes/
        #   short_pauses/
        #   long_pauses/
        
        # Process dot samples
        dot_dir = os.path.join(data_dir, 'dots')
        for file in os.listdir(dot_dir):
            if file.endswith('.wav'):
                spec = self.preprocess_audio(os.path.join(dot_dir, file))
                if spec is not None:
                    segments = self.segment_audio(spec)
                    features.extend(segments)
                    labels.extend([self.reverse_label_map['.']] * len(segments))
                    
        # Process dash samples
        dash_dir = os.path.join(data_dir, 'dashes')
        for file in os.listdir(dash_dir):
            if file.endswith('.wav'):
                spec = self.preprocess_audio(os.path.join(dash_dir, file))
                if spec is not None:
                    segments = self.segment_audio(spec)
                    features.extend(segments)
                    labels.extend([self.reverse_label_map['-']] * len(segments))
        
        # Process short pause samples
        s_pause_dir = os.path.join(data_dir, 'short_pauses')
        for file in os.listdir(s_pause_dir):
            if file.endswith('.wav'):
                spec = self.preprocess_audio(os.path.join(s_pause_dir, file))
                if spec is not None:
                    segments = self.segment_audio(spec)
                    features.extend(segments)
                    labels.extend([self.reverse_label_map['short_pause']] * len(segments))
        
        # Process long pause samples
        l_pause_dir = os.path.join(data_dir, 'long_pauses')
        for file in os.listdir(l_pause_dir):
            if file.endswith('.wav'):
                spec = self.preprocess_audio(os.path.join(l_pause_dir, file))
                if spec is not None:
                    segments = self.segment_audio(spec)
                    features.extend(segments)
                    labels.extend([self.reverse_label_map['long_pause']] * len(segments))
        
        return np.array(features), np.array(labels)
    
    def train_model(self, data_dir, batch_size=32, epochs=50, validation_split=0.2):
        """Train the model on the prepared dataset"""
        # Prepare dataset
        X, y = self.prepare_dataset(data_dir)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Get input shape from processed data
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Build model
        self.model = self.build_model(input_shape)
        
        # Define callbacks
        checkpoint_path = "morse_model_best.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor='val_accuracy', 
            verbose=1, save_best_only=True, mode='max'
        )
        early_stop = EarlyStopping(
            monitor='val