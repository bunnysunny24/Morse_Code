import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os
import json
import random
import argparse
from tqdm import tqdm
import re
import multiprocessing
from functools import partial
import time
import gc

# Skip NLTK for speed; using static sentences is faster
USE_NLTK = False
if USE_NLTK:
    try:
        import nltk
        from nltk.corpus import brown, gutenberg
        nltk.download('brown', quiet=True)
        nltk.download('gutenberg', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK corpus loaded successfully")
    except ImportError:
        print("NLTK not available, using built-in sentences only")
        USE_NLTK = False

# Constants - slightly simplified for speed
DOT_DURATION = 80  # ms (reduced from 100 for faster generation)
DASH_DURATION = 3 * DOT_DURATION
INTRA_CHAR_PAUSE = DOT_DURATION
INTER_CHAR_PAUSE = 3 * DOT_DURATION
WORD_PAUSE = 7 * DOT_DURATION
TONE_FREQ = 800  # Hz
SAMPLE_RATE = 22050  # Reduced from 44100 for faster processing

# Morse code dictionary
TEXT_TO_MORSE = {
    'A': '.-',     'B': '-...',   'C': '-.-.',   'D': '-..',
    'E': '.',      'F': '..-.',   'G': '--.',    'H': '....',
    'I': '..',     'J': '.---',   'K': '-.-',    'L': '.-..',
    'M': '--',     'N': '-.',     'O': '---',    'P': '.--.',
    'Q': '--.-',   'R': '.-.',    'S': '...',    'T': '-',
    'U': '..-',    'V': '...-',   'W': '.--',    'X': '-..-',
    'Y': '-.--',   'Z': '--..',
    '1': '.----',  '2': '..---',  '3': '...--',  '4': '....-',
    '5': '.....',  '6': '-....',  '7': '--...',  '8': '---..',
    '9': '----.',  '0': '-----',
    '.': '.-.-.-', ',': '--..--', '?': '..--..',
    '/': '-..-.',  '-': '-....-', '!': '-.-.--'
}

# Shorter collection of sentences for faster generation
SENTENCES = [
    # Short and common
    "HELLO WORLD", "SOS", "MAYDAY", "CQ DX", "HOW ARE YOU", "THANK YOU", 
    "GOOD MORNING", "GOOD NIGHT", "TESTING", "QTH", "QSL", "QRZ",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    
    # Medium length technical
    "QUANTUM PHYSICS", "MACHINE LEARNING", "ARTIFICIAL INTELLIGENCE",
    "DIGITAL SIGNAL PROCESSING", "DATA SCIENCE", "FREQUENCY MODULATION",
    "RADIO TRANSMISSION", "ELECTROMAGNETIC WAVE", "SIGNAL STRENGTH",
    
    # Morse-related
    "CQ CQ CQ DE AMATEUR RADIO", "SIGNAL REPORT", "FREQUENCY SHIFT", 
    "MORSE CODE IS FUN", "RADIO OPERATORS", "ANTENNA TUNER",
    
    # Complex structures (limited number for speed)
    "IF DATA IS PROCESSED THEN SIGNALS ARE TRANSMITTED",
    "WHEN RADIO WORKS COMMUNICATION HAPPENS QUICKLY",
    "THE SYSTEM ANALYZES WHILE THE NETWORK TRANSMITS"
]

# Domain-specific vocabulary (reduced for speed)
DOMAIN_VOCABULARIES = {
    "technology": ["ALGORITHM", "COMPUTER", "DATABASE", "ENCRYPTION", "HARDWARE", 
                  "INTERFACE", "KERNEL", "NETWORK", "PROTOCOL", "SERVER", "VIRTUAL"],
    
    "morse": ["ANTENNA", "BEACON", "CALLSIGN", "FREQUENCY", "KEYER", "OPERATOR", 
              "QRM", "SIGNAL", "TRANSMITTER", "RADIO"]
}

# Precomputed sine waves for common durations to avoid regenerating them
SINE_CACHE = {}

def generate_sine_wave(freq, duration_ms, sample_rate=SAMPLE_RATE):
    """Generate a sine wave using numpy (optimized with caching)"""
    key = (freq, duration_ms, sample_rate)
    if key in SINE_CACHE:
        return SINE_CACHE[key]
    
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    
    # Generate sine wave samples
    samples = np.sin(2 * np.pi * freq * np.arange(num_samples) / sample_rate)
    
    # Convert to 16-bit PCM
    samples = (samples * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    
    # Cache for future use (limit cache size)
    if len(SINE_CACHE) < 10:
        SINE_CACHE[key] = audio_segment
    
    return audio_segment

def generate_morse_audio(text, noise_level=0.05, speed_factor=1.0, freq_variation=0.0):
    """
    Generate Morse code audio from text (optimized version)
    
    Args:
        text: Text to convert to Morse code
        noise_level: Amount of noise to add (0 to 1)
        speed_factor: Speed multiplier (higher = faster)
        freq_variation: Amount of frequency variation (0 to 1)
        
    Returns:
        AudioSegment: The generated Morse code audio
        list: Sequence of Morse elements for labeling
    """
    # Limit text length to speed up generation
    if len(text) > 50:  # Reduced max length
        text = text[:50]
        
    # Adjust durations based on speed factor
    dot_dur = int(DOT_DURATION / speed_factor)
    dash_dur = int(DASH_DURATION / speed_factor)
    intra_char = int(INTRA_CHAR_PAUSE / speed_factor)
    inter_char = int(INTER_CHAR_PAUSE / speed_factor)
    word_pause = int(WORD_PAUSE / speed_factor)
    
    # Apply frequency variation
    base_freq = TONE_FREQ
    if freq_variation > 0:
        freq_range = TONE_FREQ * freq_variation
        base_freq = TONE_FREQ + random.uniform(-freq_range, freq_range)
    
    # Generate sound segments
    dot_sound = generate_sine_wave(base_freq, dot_dur)
    dash_sound = generate_sine_wave(base_freq, dash_dur)
    
    # Create silence segments
    intra_char_space = AudioSegment.silent(duration=intra_char)
    inter_char_space = AudioSegment.silent(duration=inter_char)
    word_space = AudioSegment.silent(duration=word_pause)
    
    # Initialize output
    audio = AudioSegment.silent(duration=0)
    elements = []  # To store the sequence of elements for labeling
    
    # Process each character
    for char in text.upper():
        if char == ' ':
            audio += word_space
            elements.append('long_pause')
        elif char in TEXT_TO_MORSE:
            morse = TEXT_TO_MORSE[char]
            for i, symbol in enumerate(morse):
                if symbol == '.':
                    audio += dot_sound
                    elements.append('dot')
                elif symbol == '-':
                    audio += dash_sound
                    elements.append('dash')
                    
                # Add intra-character space (except after last symbol)
                if i < len(morse) - 1:
                    audio += intra_char_space
                    elements.append('short_pause')
            
            # Add inter-character space
            audio += inter_char_space
            elements.append('short_pause')
    
    # Add noise if specified (simplified for speed)
    if noise_level > 0:
        try:
            # Generate noise (simpler approach)
            noise_samples = np.random.normal(0, noise_level, len(audio.raw_data) // 2).astype(np.float32)
            noise_samples = (noise_samples * 32767).astype(np.int16)
            
            noise_audio = AudioSegment(
                noise_samples.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,
                channels=1
            )
            
            # Make sure noise is the same length as audio
            if len(noise_audio) < len(audio):
                noise_audio = noise_audio + AudioSegment.silent(duration=len(audio) - len(noise_audio))
            else:
                noise_audio = noise_audio[:len(audio)]
            
            # Mix with audio
            audio = audio.overlay(noise_audio)
        except Exception as e:
            pass  # Silently ignore errors for speed
    
    return audio, elements

def generate_text_sample(base_samples):
    """Generate a text sample for Morse code conversion"""
    # Reuse a base sample with high probability
    if base_samples and random.random() < 0.7:
        # Favor shorter samples for faster processing
        samples_by_length = sorted(base_samples, key=len)
        # Get a sample from the first 70% of the sorted list (shorter samples)
        index = random.randint(0, int(len(samples_by_length) * 0.7))
        return samples_by_length[index]
    
    # Generate new text
    text_type = random.choices(
        ["word", "chars", "short_sentence"],  # Simplified options
        weights=[20, 10, 10]
    )[0]
    
    if text_type == "word":
        # Generate a single random word
        word_len = random.randint(3, 8)  # Shorter words
        return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(word_len))
    
    elif text_type == "chars":
        # Generate random characters (useful for training individual letters)
        chars = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', random.randint(1, 5))
        return ' '.join(chars)
    
    else:  # short_sentence
        # Generate a short random sentence
        word_count = random.randint(2, 5)  # Fewer words
        words = []
        for _ in range(word_count):
            word_len = random.randint(2, 6)  # Shorter words
            word = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(word_len))
            words.append(word)
        return ' '.join(words)

def process_sample(args):
    """Process a single sample (for multiprocessing)"""
    index, output_dir, text = args
    
    try:
        # Generate parameters
        speed = random.uniform(0.8, 1.3)
        noise = random.uniform(0.01, 0.15)
        freq_var = random.uniform(0.0, 0.1)
        
        # Generate audio
        audio, elements = generate_morse_audio(
            text, 
            noise_level=noise, 
            speed_factor=speed,
            freq_variation=freq_var
        )
        
        # Save audio file
        file_name = f"morse_sample_{index:05d}.wav"
        file_path = os.path.join(output_dir, file_name)
        audio.export(file_path, format="wav")
        
        # Return information for labels
        return file_name, elements, text
    except Exception as e:
        return None, None, None

def generate_dataset_parallel(output_dir, num_samples=1000, start_index=0, num_processes=None):
    """
    Generate a dataset of Morse code audio files with labels using parallel processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_processes} processes for parallel generation")
    
    # Load existing labels if any
    labels_file = os.path.join(output_dir, 'morse_labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            try:
                labels = json.load(f)
                print(f"Loaded existing labels for {len(labels)} samples")
            except:
                labels = {}
    else:
        labels = {}
    
    # Prepare base text samples
    base_samples = SENTENCES.copy()
    
    # Add domain vocabulary
    for domain, words in DOMAIN_VOCABULARIES.items():
        base_samples.extend(words)
    
    # Add single letters and numbers
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        base_samples.append(char)
    
    # Generate text samples in advance (faster than generating during processing)
    print("Preparing text samples...")
    text_samples = []
    for i in range(start_index, start_index + num_samples):
        if i < len(base_samples):
            text = base_samples[i - start_index]
        else:
            text = generate_text_sample(base_samples)
        text_samples.append(text)
    
    # Create batch arguments
    batch_args = [(i, output_dir, text) for i, text in zip(range(start_index, start_index + num_samples), text_samples)]
    
    print(f"Generating {num_samples} samples starting from index {start_index}...")
    start_time = time.time()
    
    # Process batches using multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_sample, batch_args),
            total=num_samples,
            desc="Generating samples"
        ))
    
    # Process results
    new_labels = {}
    sample_texts = {}
    for file_name, elements, text in results:
        if file_name is not None:
            new_labels[file_name] = elements
            sample_texts[file_name] = text
    
    # Update labels
    labels.update(new_labels)
    
    # Save labels to JSON file
    with open(labels_file, 'w') as f:
        json.dump(labels, f)
    
    # Save text samples for reference
    with open(os.path.join(output_dir, 'sample_texts.json'), 'w') as f:
        json.dump(sample_texts, f)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Dataset generation complete. {len(new_labels)} samples created in {elapsed:.1f} seconds")
    print(f"Average time per sample: {elapsed / num_samples:.3f} seconds")
    
    # Generate some statistics
    print("\nDataset Statistics:")
    total_elements = sum(len(elements) for elements in labels.values())
    element_counts = {element: 0 for element in ["dot", "dash", "short_pause", "long_pause"]}
    for elements in labels.values():
        for element in elements:
            if element in element_counts:
                element_counts[element] += 1
    
    print(f"Total Morse elements: {total_elements}")
    for element, count in element_counts.items():
        print(f"  {element}: {count} ({count/total_elements*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Morse code audio dataset (fast parallel version)')
    parser.add_argument('--output', default='./morse_audio_dataset', help='Output directory for dataset')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--start', type=int, default=0, help='Starting index for sample filenames')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--simple', action='store_true', help='Use simplified generation for even faster processing')
    
    args = parser.parse_args()
    
    print(f"Starting fast Morse code dataset generator...")
    print(f"Target: {args.samples} samples in '{args.output}'")
    
    # Run with multiprocessing
    generate_dataset_parallel(
        args.output, 
        num_samples=args.samples, 
        start_index=args.start,
        num_processes=args.processes
    )