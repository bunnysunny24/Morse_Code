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
import h5py  # For efficient storage of large arrays

# Create the directory for saved models if it doesn't exist
save_dir = r"D:\Bunny\MorseCode\backend\scipts\saved_models"
os.makedirs(save_dir, exist_ok=True)

# Create a directory for temporary storage
temp_dir = os.path.join(save_dir, "temp_data")
os.makedirs(temp_dir, exist_ok=True)

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # max duration in seconds to analyze
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MORSE_ELEMENTS = ['dot', 'dash', 'short_pause', 'long_pause']
BATCH_SIZE_PROCESSING = 1000  # Process data in smaller batches to reduce memory usage

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
    
    # If fixed_time_steps is provided, resize the spectrogram
    if fixed_time_steps is not None and mel_spec_db.shape[1] > fixed_time_steps:
        # Trim if longer than fixed_time_steps
        mel_spec_db = mel_spec_db[:, :fixed_time_steps]
    
    return mel_spec_db

def find_max_time_steps(features_list):
    """Find the maximum time steps across all features"""
    max_time = 0
    for feat in features_list:
        if feat.shape[1] > max_time:
            max_time = feat.shape[1]
    return max_time

def process_batch_and_save(features_batch, labels_batch, batch_idx, max_time_steps, output_file):
    """Process a batch of features and save to HDF5"""
    standardized_batch = []
    
    for feat in tqdm(features_batch, desc=f"Processing batch {batch_idx}"):
        # Current shape and dimensions
        freq_bins, time_steps = feat.shape
        
        # Create padded array
        padded_feat = np.zeros((freq_bins, max_time_steps))
        
        # Copy original data
        padded_feat[:, :time_steps] = feat
        
        # Add channel dimension for Conv2D
        padded_feat = np.expand_dims(padded_feat, axis=-1)
        standardized_batch.append(padded_feat)
    
    # Convert to array
    standardized_batch = np.array(standardized_batch)
    
    # Save batch to HDF5 file
    with h5py.File(output_file, 'a') as f:
        # Create dataset for this batch if it doesn't exist
        if 'features' not in f:
            f.create_dataset('features', data=standardized_batch, maxshape=(None, *standardized_batch.shape[1:]), 
                           chunks=True, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=np.array(labels_batch), maxshape=(None, len(labels_batch[0])), 
                           chunks=True)
        else:
            # Append to existing dataset
            f['features'].resize((f['features'].shape[0] + standardized_batch.shape[0]), axis=0)
            f['features'][-standardized_batch.shape[0]:] = standardized_batch
            
            f['labels'].resize((f['labels'].shape[0] + len(labels_batch)), axis=0)
            f['labels'][-len(labels_batch):] = np.array(labels_batch)
    
    # Clean up memory
    del standardized_batch
    gc.collect()

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

def process_and_save_dataset(dataset_path, label_file, output_file, max_samples=None, sample_limit_per_class=20000):
    """
    Process dataset and save to HDF5 file to avoid memory issues
    
    Args:
        dataset_path: Path to directory containing audio files
        label_file: Path to JSON file containing labels
        output_file: Path to output HDF5 file
        max_samples: Maximum number of samples to load (useful for testing)
        sample_limit_per_class: Maximum number of samples per class to balance the dataset
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
    
    # Count and limit samples per class to prevent imbalance
    class_counts = {element: 0 for element in MORSE_ELEMENTS}
    filtered_files = []
    
    for file_name in file_names:
        if not labels[file_name]:  # Skip files with empty labels
            continue
            
        morse_element = labels[file_name][0]  # Use first element as label
        if morse_element in MORSE_ELEMENTS and class_counts[morse_element] < sample_limit_per_class:
            filtered_files.append(file_name)
            class_counts[morse_element] += 1
    
    print(f"Using {len(filtered_files)} files after balancing classes")
    for element, count in class_counts.items():
        print(f"  {element}: {count} samples")
    
    # Process data in batches to reduce memory usage
    batch_size = BATCH_SIZE_PROCESSING
    
    # First, find the max time steps across a subset of samples
    # This avoids loading all samples just to find the maximum
    time_steps_sample = min(1000, len(filtered_files))
    print(f"Analyzing {time_steps_sample} samples to find max time steps...")
    
    features_sample = []
    for file_name in tqdm(filtered_files[:time_steps_sample], desc="Analyzing sample spectrograms"):
        file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_path):
            continue
        
        try:
            features = load_and_preprocess_audio(file_path)
            features_sample.append(features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    max_time_steps = find_max_time_steps(features_sample)
    print(f"Maximum time steps across sample: {max_time_steps}")
    
    # Clear memory
    del features_sample
    gc.collect()
    
    # Process all files in batches
    for batch_idx in range(0, len(filtered_files), batch_size):
        batch_files = filtered_files[batch_idx:batch_idx + batch_size]
        
        features_batch = []
        labels_batch = []
        
        for file_name in tqdm(batch_files, desc=f"Loading batch {batch_idx//batch_size + 1}/{(len(filtered_files)-1)//batch_size + 1}"):
            file_path = os.path.join(dataset_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping.")
                continue
            
            try:
                # Extract features
                features = load_and_preprocess_audio(file_path)
                features_batch.append(features)
                
                # Get label
                morse_element = labels[file_name][0] if labels[file_name] else "dot"
                
                # Convert to one-hot encoding
                one_hot = [0] * len(MORSE_ELEMENTS)
                try:
                    one_hot[MORSE_ELEMENTS.index(morse_element)] = 1
                    labels_batch.append(one_hot)
                except ValueError:
                    print(f"Warning: Unknown element '{morse_element}' in {file_name}, using 'dot' instead")
                    one_hot[MORSE_ELEMENTS.index("dot")] = 1
                    labels_batch.append(one_hot)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Process and save this batch
        process_batch_and_save(features_batch, labels_batch, batch_idx//batch_size + 1, max_time_steps, output_file)
        
        # Clear memory
        del features_batch, labels_batch
        gc.collect()
    
    # Print dataset info
    with h5py.File(output_file, 'r') as f:
        print(f"Final dataset size: {f['features'].shape}")
    
    return max_time_steps

def train_model(model, dataset_file, batch_size=32, epochs=50, model_path='morse_recognition_model.h5'):
    """
    Train model with early stopping and save a checkpoint after each epoch
    
    Args:
        model: Keras model
        dataset_file: Path to HDF5 file containing features and labels
        batch_size: Batch size for training
        epochs: Number of epochs for training
        model_path: Path to save best model
    """
    # Get directory from model_path
    model_dir = os.path.dirname(model_path)
    latest_model_path = os.path.join(model_dir, 'latest_model_v2.h5')
    
    # Create a directory for epoch checkpoints
    checkpoints_dir = os.path.join(model_dir, 'epoch_checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Epoch checkpoints will be saved to: {checkpoints_dir}")
    
    # Load dataset size info
    with h5py.File(dataset_file, 'r') as f:
        num_samples = f['features'].shape[0]
    
    # Create data generators that load from HDF5 to avoid memory issues
    class HDF5DataGenerator(keras.utils.Sequence):
        def __init__(self, h5_file, indices, batch_size=32):
            self.h5_file = h5_file
            self.indices = indices
            self.batch_size = batch_size
            
        def __len__(self):
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
        
        def __getitem__(self, idx):
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            
            with h5py.File(self.h5_file, 'r') as f:
                batch_features = f['features'][batch_indices]
                batch_labels = f['labels'][batch_indices]
                
            return batch_features, batch_labels
    
    # Split into training and validation sets
    train_split = 0.8
    indices = np.random.permutation(num_samples)
    train_idx = indices[:int(train_split * num_samples)]
    val_idx = indices[int(train_split * num_samples):]
    
    train_generator = HDF5DataGenerator(dataset_file, train_idx, batch_size)
    val_generator = HDF5DataGenerator(dataset_file, val_idx, batch_size)
    
    print(f"Training with {len(train_idx)} samples, validating with {len(val_idx)} samples")
    
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
        ),
        # Save a separate checkpoint for each epoch
        keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoints_dir, 'morse_model_epoch_{epoch:03d}.h5'),
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=True,
        workers=4
    )
    
    print(f"Epoch checkpoints saved to: {checkpoints_dir}")
    return model, history

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

def predict_from_audio(model, audio_file, max_time_steps):
    """
    Predict Morse code elements from audio file and decode to text
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
    parser.add_argument('--sample_limit', type=int, default=20000, help='Maximum samples per class (for balance)')
    
    args = parser.parse_args()
    
    # Determine labels file path if not specified
    if args.labels is None:
        args.labels = os.path.join(args.dataset, 'morse_labels.json')
    
    print(f"Using dataset: {args.dataset}")
    print(f"Using labels: {args.labels}")
    print(f"Model will be saved to: {args.model_path}")
    print(f"Latest model will be saved as: {os.path.join(save_dir, 'latest_model_v2.h5')}")
    print(f"Per-epoch checkpoints will be saved in: {os.path.join(save_dir, 'epoch_checkpoints')}")
    
    # Check if dataset and labels exist
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory {args.dataset} not found.")
        return
    
    if not os.path.exists(args.labels):
        print(f"Error: Labels file {args.labels} not found.")
        return
    
    # Path for processed dataset
    h5_dataset = os.path.join(temp_dir, 'morse_dataset.h5')
    
    # Process and save dataset
    print("Processing dataset and saving to HDF5...")
    max_time_steps = process_and_save_dataset(
        args.dataset, 
        args.labels, 
        h5_dataset, 
        max_samples=args.max_samples,
        sample_limit_per_class=args.sample_limit
    )
    
    # Get input shape for model
    with h5py.File(h5_dataset, 'r') as f:
        input_shape = f['features'][0].shape
        
    print(f"Input shape for model: {input_shape}")
    
    # Create the model
    model = create_model(input_shape, len(MORSE_ELEMENTS))
    model.summary()
    
    # Train the model
    print(f"Training model with batch size {args.batch_size} for {args.epochs} epochs...")
    model, history = train_model(
        model, 
        h5_dataset,
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
    
    # Set memory growth for GPU if available
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {len(gpus)} GPUs")
    except:
        print("No GPU available or could not configure memory growth")
    
    main()