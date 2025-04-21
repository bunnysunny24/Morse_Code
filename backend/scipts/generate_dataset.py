import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os
import json
import random

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
    '/': '-..-.',  '-': '-....-', '!': '-.-.--'
}

def generate_morse_audio(text, noise_level=0.05, speed_factor=1.0):
    """
    Generate Morse code audio from text
    
    Args:
        text: Text to convert to Morse code
        noise_level: Amount of noise to add (0 to 1)
        speed_factor: Speed multiplier (higher = faster)
        
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
    
    # Generate sine wave tones
    dot_sound = Sine(TONE_FREQ).to_audio_segment(duration=dot_dur)
    dash_sound = Sine(TONE_FREQ).to_audio_segment(duration=dash_dur)
    
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
        noise = AudioSegment.silent(duration=len(audio))
        noise = noise.overlay(AudioSegment.from_file_using_temporary_files(
            np.random.normal(0, noise_level, len(audio) * audio.frame_rate).astype(np.float32).tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=1
        ))
        audio = audio.overlay(noise)
    
    return audio, elements

def generate_dataset(output_dir, num_samples=100, text_samples=None):
    """
    Generate a dataset of Morse code audio files with labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
            "HOW ARE YOU"
        ]
    
    labels = {}
    
    for i in range(num_samples):
        if i < len(text_samples):
            text = text_samples[i]
        else:
            # Generate random text for additional samples
            word_count = random.randint(1, 5)
            words = []
            for _ in range(word_count):
                word_len = random.randint(1, 8)
                word = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(word_len))
                words.append(word)
            text = ' '.join(words)
        
        # Randomize speed and noise level for variety
        speed = random.uniform(0.8, 1.2)
        noise = random.uniform(0.01, 0.1)
        
        # Generate audio
        audio, elements = generate_morse_audio(text, noise_level=noise, speed_factor=speed)
        
        # Save audio file
        file_name = f"morse_sample_{i:04d}.wav"
        file_path = os.path.join(output_dir, file_name)
        audio.export(file_path, format="wav")
        
        # Store labels
        labels[file_name] = elements
        
        print(f"Generated {file_name}: '{text}'")
    
    # Save labels to JSON file
    with open(os.path.join(output_dir, 'morse_labels.json'), 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Dataset generation complete. {num_samples} samples created in {output_dir}")

if __name__ == "__main__":
    generate_dataset("./morse_audio_dataset", num_samples=200)