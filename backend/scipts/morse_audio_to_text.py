import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # max duration in seconds to analyze
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
    '----.': '9',  '-----': '0'
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

def load_dataset(dataset_path, label_file):
    """
    Load dataset from directory
    
    Args:
        dataset_path: Path to directory containing audio files
        label_file: Path to JSON file containing labels
        
    Returns:
        X: Features
        y: Labels
    """
    # Load labels from JSON file
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    X = []
    y = []
    
    for file_name, morse_elements in labels.items():
        file_path = os.path.join(dataset_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        
        try:
            # Extract features
            features = load_and_preprocess_audio(file_path)
            X.append(features)
            
            # Convert labels to one-hot encoding
            label_seq = []
            for element in morse_elements:
                one_hot = [0] * len(MORSE_ELEMENTS)
                one_hot[MORSE_ELEMENTS.index(element)] = 1
                label_seq.append(one_hot)
            
            y.append(label_seq)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
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
            'morse_recognition_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
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
        Decoded text
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
    decoded_text = decode_morse_sequence(morse_sequence)
    
    return decoded_text

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
    # Path to your dataset - REPLACE WITH YOUR ACTUAL PATHS
    dataset_path = "./morse_audio_dataset/"
    label_file = "./morse_labels.json"
    
    print("Loading and preprocessing dataset...")
    X, y = load_dataset(dataset_path, label_file)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset loaded. Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create the model
    input_shape = X_train[0].shape
    model = create_model(input_shape, len(MORSE_ELEMENTS))
    model.summary()
    
    # Train the model
    print("Training model...")
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model.save('morse_recognition_model.h5')
    print("Model saved as 'morse_recognition_model.h5'")
    
    # Example prediction
    test_file = input("Enter path to a Morse code audio file to test (or press Enter to skip): ")
    if test_file and os.path.exists(test_file):
        decoded_text = predict_from_audio(model, test_file)
        print(f"Decoded text: {decoded_text}")

if __name__ == "__main__":
    main()