import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import gc  # Garbage collection for memory management

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # max duration in seconds to analyze
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MORSE_ELEMENTS = ['dot', 'dash', 'short_pause', 'long_pause']

# Morse code dictionary for decoding (expanded with punctuation)
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

def load_and_preprocess_audio(file_path):
    """
    Load and preprocess audio file to extract mel spectrogram features
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Mel spectrogram features
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

def create_model(input_shape, num_classes=4):
    """
    Create a CNN+LSTM model for audio classification
    
    Args:
        input_shape: Shape of input features (mel spectrogram)
        num_classes: Number of classes to predict
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional layers to extract features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Reshape for LSTM
    x = layers.Reshape((-1, x.shape[-1]))(x)
    
    # LSTM layers for sequential information
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_dataset(dataset_path, label_file, max_samples=None, batch_processing=False):
    """
    Load dataset from directory
    
    Args:
        dataset_path: Path to directory containing audio files
        label_file: Path to JSON file containing labels
        max_samples: Maximum number of samples to load (useful for testing)
        batch_processing: Whether to process in batches to save memory
        
    Returns:
        X: Features
        y: Labels
    """
    # Load labels from JSON file
    print(f"Loading labels from {label_file}")
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    # Get list of available files
    file_names = list(labels.keys())
    
    if max_samples and max_samples < len(file_names):
        print(f"Limiting to {max_samples} samples")
        file_names = file_names[:max_samples]
    
    total_files = len(file_names)
    print(f"Found {total_files} labeled files")
    
    if batch_processing:
        # Return a generator to process files in batches
        def dataset_generator():
            for file_name in tqdm(file_names, desc="Processing audio files"):
                file_path = os.path.join(dataset_path, file_name)
                if not os.path.exists(file_path):
                    continue
                
                try:
                    # Extract features
                    features = load_and_preprocess_audio(file_path)
                    
                    # Convert labels to one-hot encoding
                    morse_elements = labels[file_name]
                    label_seq = []
                    for element in morse_elements:
                        one_hot = [0] * len(MORSE_ELEMENTS)
                        try:
                            one_hot[MORSE_ELEMENTS.index(element)] = 1
                            label_seq.append(one_hot)
                        except ValueError:
                            print(f"Warning: Unknown element '{element}' in {file_name}, skipping")
                    
                    yield features, np.array(label_seq)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return dataset_generator, total_files
    else:
        # Process all files at once
        X = []
        y = []
        
        for file_name in tqdm(file_names, desc="Processing audio files"):
            file_path = os.path.join(dataset_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping.")
                continue
            
            try:
                # Extract features
                features = load_and_preprocess_audio(file_path)
                X.append(features)
                
                # Convert labels to one-hot encoding
                morse_elements = labels[file_name]
                label_seq = []
                for element in morse_elements:
                    one_hot = [0] * len(MORSE_ELEMENTS)
                    try:
                        one_hot[MORSE_ELEMENTS.index(element)] = 1
                        label_seq.append(one_hot)
                    except ValueError:
                        print(f"Warning: Unknown element '{element}' in {file_name}, skipping")
                
                y.append(label_seq)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return np.array(X), np.array(y)

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, model_path='morse_recognition_model.h5'):
    """
    Train model with early stopping
    """
    # Define callbacks for training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Save model after each epoch to prevent data loss
        keras.callbacks.ModelCheckpoint(
            'morse_model_latest.h5',
            save_best_only=False,
            verbose=0
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def decode_morse_sequence(sequence):
    """
    Decode a sequence of Morse elements into text
    
    Args:
        sequence: List of Morse elements ('dot', 'dash', 'short_pause', 'long_pause')
        
    Returns:
        Decoded text and the morse code representation
    """
    current_char = []
    message = []
    morse_code = ""
    
    for element in sequence:
        if element == 'dot':
            current_char.append('.')
            morse_code += '.'
        elif element == 'dash':
            current_char.append('-')
            morse_code += '-'
        elif element == 'short_pause' and current_char:
            # End of character
            morse_char = ''.join(current_char)
            if morse_char in MORSE_TO_TEXT:
                message.append(MORSE_TO_TEXT[morse_char])
            current_char = []
            morse_code += ' '
        elif element == 'long_pause':
            # End of word
            if current_char:
                morse_char = ''.join(current_char)
                if morse_char in MORSE_TO_TEXT:
                    message.append(MORSE_TO_TEXT[morse_char])
                current_char = []
            message.append(' ')
            morse_code += ' / '
    
    # Handle any remaining character
    if current_char:
        morse_char = ''.join(current_char)
        if morse_char in MORSE_TO_TEXT:
            message.append(MORSE_TO_TEXT[morse_char])
    
    return ''.join(message).strip(), morse_code.strip()

def predict_from_audio(model, audio_file):
    """
    Predict Morse code elements from audio file and decode to text
    
    Args:
        model: Trained Keras model
        audio_file: Path to audio file
        
    Returns:
        Predicted text
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
    decoded_text, morse_code = decode_morse_sequence(morse_sequence)
    
    return decoded_text, morse_code

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """
    Main function to train model and make predictions
    """
    parser = argparse.ArgumentParser(description='Train Morse code recognition model')
    parser.add_argument('--dataset', default='./morse_audio_dataset/', help='Path to dataset directory')
    parser.add_argument('--labels', help='Path to labels file (JSON)')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to load')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--model_path', default='morse_recognition_model.h5', help='Path to save model')
    parser.add_argument('--test_file', help='Audio file to test after training')
    
    args = parser.parse_args()
    
    # Determine labels file path if not specified
    if args.labels is None:
        args.labels = os.path.join(args.dataset, 'morse_labels.json')
    
    print(f"Using dataset: {args.dataset}")
    print(f"Using labels: {args.labels}")
    
    # Check if dataset and labels exist
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory {args.dataset} not found.")
        return
    
    if not os.path.exists(args.labels):
        print(f"Error: Labels file {args.labels} not found.")
        return
    
    print("Loading and preprocessing dataset...")
    X, y = load_dataset(args.dataset, args.labels, max_samples=args.max_samples)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset loaded. Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create the model
    input_shape = X_train[0].shape
    model = create_model(input_shape, len(MORSE_ELEMENTS))
    model.summary()
    
    # Train the model
    print(f"Training model with batch size {args.batch_size} for {args.epochs} epochs...")
    model, history = train_model(
        model, X_train, y_train, X_val, y_val, 
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_path=args.model_path
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model.save(args.model_path)
    print(f"Model saved as '{args.model_path}'")
    
    # Example prediction
    test_file = args.test_file
    if test_file is None:
        test_file = input("Enter path to a Morse code audio file to test (or press Enter to skip): ")
    
    if test_file and os.path.exists(test_file):
        decoded_text, morse_code = predict_from_audio(model, test_file)
        print("\nResults:")
        print(f"Decoded Morse code: {morse_code}")
        print(f"Decoded text: {decoded_text}")

if __name__ == "__main__":
    main()