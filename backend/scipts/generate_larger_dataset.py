import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os
import json
import random
import argparse
from tqdm import tqdm

# Constants
DOT_DURATION = 100  # ms
DASH_DURATION = 3 * DOT_DURATION
INTRA_CHAR_PAUSE = DOT_DURATION
INTER_CHAR_PAUSE = 3 * DOT_DURATION
WORD_PAUSE = 7 * DOT_DURATION
TONE_FREQ = 800  # Hz
SAMPLE_RATE = 44100

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
    '/': '-..-.',  '-': '-....-', '!': '-.-.--',
    '@': '.--.-.',  ':': '---...', ';': '-.-.-.',
    '=': '-...-',    '+': '.-.-.',  '&': '.-...'
}

def generate_sine_wave(freq, duration_ms, sample_rate=44100):
    """Generate a sine wave using numpy (alternative to pydub.generators.Sine)"""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    
    # Generate sine wave samples
    samples = np.sin(2 * np.pi * freq * np.arange(num_samples) / sample_rate)
    
    # Convert to 16-bit PCM
    samples = (samples * 32767).astype(np.int16)
    
    # Create AudioSegment
    return AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

def generate_morse_audio(text, noise_level=0.05, speed_factor=1.0, freq_variation=0.0):
    """
    Generate Morse code audio from text
    
    Args:
        text: Text to convert to Morse code
        noise_level: Amount of noise to add (0 to 1)
        speed_factor: Speed multiplier (higher = faster)
        freq_variation: Amount of frequency variation (0 to 1)
        
    Returns:
        AudioSegment: The generated Morse code audio
        list: Sequence of Morse elements for labeling
    """
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
    
    # Try using built-in Sine generator first, fall back to custom function if it fails
    try:
        dot_sound = Sine(base_freq).to_audio_segment(duration=dot_dur)
        dash_sound = Sine(base_freq).to_audio_segment(duration=dash_dur)
    except Exception:
        # Fallback to custom sine wave generator
        dot_sound = generate_sine_wave(base_freq, dot_dur)
        dash_sound = generate_sine_wave(base_freq, dash_dur)
    
    # Create silence segments
    intra_char_space = AudioSegment.silent(duration=intra_char)
    inter_char_space = AudioSegment.silent(duration=inter_char)
    word_space = AudioSegment.silent(duration=word_pause)
    
    # Initialize output
    audio = AudioSegment.silent(duration=0)
    elements = []  # To store the sequence of elements for labeling
    
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
    
    # Add noise if specified
    if noise_level > 0:
        try:
            # Generate noise
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
            print(f"Warning: Could not add noise to audio: {e}")
    
    return audio, elements

def generate_dataset(output_dir, num_samples=1000, text_samples=None, start_index=0, verbose=True):
    """
    Generate a dataset of Morse code audio files with labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    if text_samples is None:
        # Common English phrases and sentences for training
        text_samples = [
            "HELLO WORLD",
            "SOS",
            "MAYDAY",
            "CQ DX",
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
            "I LOVE MORSE CODE",
            "TESTING ONE TWO THREE",
            "PYTHON PROGRAMMING",
            "ARTIFICIAL INTELLIGENCE",
            "MACHINE LEARNING",
            "DEEP LEARNING",
            "HOW ARE YOU",
            "THANK YOU",
            "GOOD MORNING",
            "GOOD AFTERNOON",
            "GOOD EVENING",
            "THIS IS A TEST",
            "PLEASE REPLY",
            "OVER AND OUT",
            "ROGER THAT",
            "COPY THAT",
            "AFFIRMATIVE",
            "NEGATIVE",
            "HELP ME",
            "EMERGENCY",
            "REPEATING MESSAGE",
            "FREQUENCY CHANGE",
            "STANDING BY",
            "END OF MESSAGE",
            "BREAK BREAK",
            "QTH LOCATION",
            "QSL CONFIRM",
            "73 BEST WISHES"
        ]
        
        # Add common words
        common_words = [
            "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I", 
            "IT", "FOR", "NOT", "ON", "WITH", "HE", "AS", "YOU", "DO", "AT",
            "THIS", "BUT", "HIS", "BY", "FROM", "THEY", "WE", "SAY", "HER", "SHE",
            "OR", "AN", "WILL", "MY", "ONE", "ALL", "WOULD", "THERE", "THEIR", "WHAT"
        ]
        text_samples.extend(common_words)
        
        # Add single letters and numbers
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            text_samples.append(char)
    
    # Create progress bar if verbose
    if verbose:
        pbar = tqdm(total=num_samples, desc="Generating samples")
    
    batch_size = 50  # Save labels every batch_size samples
    
    for i in range(start_index, start_index + num_samples):
        try:
            if i < len(text_samples):
                text = text_samples[i]
            else:
                # Generate random text for additional samples
                text_type = random.choice(["word", "sentence", "characters"])
                
                if text_type == "word":
                    # Generate a single random word
                    word_len = random.randint(3, 10)
                    text = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(word_len))
                
                elif text_type == "sentence":
                    # Generate a random sentence
                    word_count = random.randint(2, 7)
                    words = []
                    for _ in range(word_count):
                        word_len = random.randint(2, 8)
                        word = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(word_len))
                        words.append(word)
                    text = ' '.join(words)
                
                else:  # characters
                    # Generate random characters (useful for training individual letters)
                    chars = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', random.randint(1, 5))
                    text = ' '.join(chars)
            
            # Randomize parameters for variety
            speed = random.uniform(0.7, 1.3)  # Wider speed range
            noise = random.uniform(0.0, 0.15)  # More variation in noise
            freq_var = random.uniform(0.0, 0.1)  # Add some frequency variation
            
            # Generate audio
            audio, elements = generate_morse_audio(
                text, 
                noise_level=noise, 
                speed_factor=speed,
                freq_variation=freq_var
            )
            
            # Save audio file
            file_name = f"morse_sample_{i:05d}.wav"
            file_path = os.path.join(output_dir, file_name)
            audio.export(file_path, format="wav")
            
            # Store labels
            labels[file_name] = elements
            
            if verbose:
                pbar.update(1)
                pbar.set_postfix({"text": text[:20] + "..." if len(text) > 20 else text})
            elif i % 10 == 0:
                print(f"Generated {i-start_index+1}/{num_samples} samples")
                
            # Save labels periodically to avoid data loss
            if i % batch_size == 0 and i > start_index:
                with open(labels_file, 'w') as f:
                    json.dump(labels, f)
        
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
    
    # Close progress bar
    if verbose:
        pbar.close()
    
    # Save labels to JSON file
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Dataset generation complete. {num_samples} samples created in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Morse code audio dataset')
    parser.add_argument('--output', default='./morse_audio_dataset', help='Output directory for dataset')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--start', type=int, default=0, help='Starting index for sample filenames')
    
    args = parser.parse_args()
    
    generate_dataset(args.output, num_samples=args.samples, start_index=args.start)