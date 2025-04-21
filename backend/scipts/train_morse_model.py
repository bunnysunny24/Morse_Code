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
import datetime

# Create the directory for saved models if it doesn't exist
save_dir = r"D:\Bunny\MorseCode\backend\scipts\saved_models"
os.makedirs(save_dir, exist_ok=True)

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

def load_and_preprocess_audio(file_path, fixed_time_steps=None):
    """
    Load and preprocess audio file to extract mel spectrogram features
    
    Args:
        file_path: Path to the audio file
        fixed_time_steps: Fixed number of time steps to resize to (if None, keep original)
        
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

def standardize_feature_shape(features_list):
    """
    Standardize the shape of all feature arrays to match the max time dimension
    
    Args:
        features_list: List of features, each with shape [freq_bins, time_steps]
        
    Returns:
        List of standardized features, each with shape [freq_bins, max_time_steps]
        Max time steps value
    """
    # Find max time dimension across all features
    max_time_steps = max(feat.shape[1] for feat in features_list)
    print(f"Maximum time steps across all spectrograms: {max_time_steps}")
    
    # Standardize all features to have the same time dimension through padding
    standardized_features = []
    
    for feat in tqdm(features_list, desc="Padding spectrograms"):
        # Current shape and dimensions
        freq_bins, time_steps = feat.shape
        
        # Create padded array
        padded_feat = np.zeros((freq_bins, max_time_steps))
        
        # Copy original data
        padded_feat[:, :time_steps] = feat
        
        standardized_features.append(padded_feat)
    
    return standardized_features, max_time_steps

def create_model(input_shape, num_classes=4):
    """
    Create a CNN model for audio classification
    
    Args:
        input_shape: Shape of input features (mel spectrogram)
        num_classes: Number of classes to predict
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_dataset(dataset_path, label_file, max_samples=None):
    """
    Load dataset from directory
    
    Args:
        dataset_path: Path to directory containing audio files
        label_file: Path to JSON file containing labels
        max_samples: Maximum number of samples to load (useful for testing)
        
    Returns:
        features_list: List of audio features
        labels_list: List of one-hot encoded labels
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
    
    # Process all files
    features_list = []
    labels_list = []
    
    for file_name in tqdm(file_names, desc="Processing audio files"):
        file_path = os.path.join(dataset_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        
        try:
            # Extract features
            features = load_and_preprocess_audio(file_path)
            features_list.append(features)
            
            # For simplicity, we'll just use the first element in the sequence as label
            morse_element = labels[file_name][0] if labels[file_name] else "dot"  # Default to dot if empty
            
            # Convert to one-hot encoding
            one_hot = [0] * len(MORSE_ELEMENTS)
            try:
                one_hot[MORSE_ELEMENTS.index(morse_element)] = 1
                labels_list.append(one_hot)
            except ValueError:
                print(f"Warning: Unknown element '{morse_element}' in {file_name}, using 'dot' instead")
                one_hot[MORSE_ELEMENTS.index("dot")] = 1
                labels_list.append(one_hot)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return features_list, labels_list

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, model_path='morse_recognition_model.h5'):
    """
    Train model with early stopping
    """
    # Get directory from model_path
    model_dir = os.path.dirname(model_path)
    latest_model_path = os.path.join(model_dir, 'latest_model_v2.h5')
    
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
            latest_model_path,
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

def predict_from_audio(model, audio_file, max_time_steps):
    """
    Predict Morse code elements from audio file and decode to text
    
    Args:
        model: Trained Keras model
        audio_file: Path to audio file
        max_time_steps: Maximum time steps for padding
        
    Returns:
        Predicted Morse element
    """
    # Load and preprocess audio
    features = load_and_preprocess_audio(audio_file)
    
    # Standardize shape
    freq_bins, time_steps = features.shape
    padded_feat = np.zeros((freq_bins, max_time_steps))
    padded_feat[:, :time_steps] = features
    
    # Add channel dimension for Conv2D and batch dimension
    features_reshaped = np.expand_dims(np.expand_dims(padded_feat, axis=0), axis=-1)
    
    # Make prediction
    predictions = model.predict(features_reshaped)
    
    # Get highest probability class
    predicted_idx = np.argmax(predictions[0])
    predicted_element = MORSE_ELEMENTS[predicted_idx]
    
    return predicted_element

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
    
    # Save the plot to the same directory as the model
    plots_dir = save_dir
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'training_history_v2.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
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
    parser.add_argument('--model_path', default=os.path.join(save_dir, 'morse_recognition_model_v2.h5'), 
                       help='Path to save model')
    parser.add_argument('--test_file', help='Audio file to test after training')
    
    args = parser.parse_args()
    
    # Determine labels file path if not specified
    if args.labels is None:
        args.labels = os.path.join(args.dataset, 'morse_labels.json')
    
    print(f"Using dataset: {args.dataset}")
    print(f"Using labels: {args.labels}")
    print(f"Model will be saved to: {args.model_path}")
    print(f"Latest model will be saved as: {os.path.join(save_dir, 'latest_model_v2.h5')}")
    
    # Check if dataset and labels exist
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory {args.dataset} not found.")
        return
    
    if not os.path.exists(args.labels):
        print(f"Error: Labels file {args.labels} not found.")
        return
    
    print("Loading and preprocessing dataset...")
    features_list, labels_list = load_dataset(args.dataset, args.labels, max_samples=args.max_samples)
    
    # Standardize feature shapes
    standardized_features, max_time_steps = standardize_feature_shape(features_list)
    
    # Add channel dimension for Conv2D
    X = np.array([np.expand_dims(feat, axis=-1) for feat in standardized_features])
    y = np.array(labels_list)
    
    print(f"Final dataset shapes - X: {X.shape}, y: {y.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset loaded. Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create the model
    input_shape = X_train[0].shape
    print(f"Input shape for model: {input_shape}")
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
    
    # Save the final model as latest_model_v2.h5
    final_model_path = os.path.join(save_dir, 'latest_model_v2.h5')
    model.save(final_model_path)
    print(f"Final model saved as '{final_model_path}'")
    
    # Example prediction
    test_file = args.test_file
    if test_file is None:
        test_file = input("Enter path to a Morse code audio file to test (or press Enter to skip): ")
    
    if test_file and os.path.exists(test_file):
        predicted_element = predict_from_audio(model, test_file, max_time_steps)
        print("\nResults:")
        print(f"Predicted Morse element: {predicted_element}")
        print("Note: This simplified model only predicts the dominant element in the audio.")
        print("For more complete Morse code processing, window-based analysis would be needed.")

if __name__ == "__main__":
    # Print current user and time information
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        print(f"Current User's Login: {os.getlogin()}")
    except Exception:
        print("Could not determine current user login")
    
    main()