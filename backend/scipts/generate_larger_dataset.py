import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os
import json
import random
import argparse
from tqdm import tqdm
import re
from nltk.corpus import brown, gutenberg
import nltk

# Try to download NLTK data if not present
try:
    nltk.download('brown', quiet=True)
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Note: NLTK data download failed. Will use built-in sentences only.")

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

# Comprehensive collection of advanced sentences
ADVANCED_SENTENCES = [
    # Scientific and technical
    "QUANTUM MECHANICS DESCRIBES NATURE AT THE SUBATOMIC SCALE",
    "THE FIBONACCI SEQUENCE APPEARS THROUGHOUT NATURE",
    "ARTIFICIAL NEURAL NETWORKS MIMIC BIOLOGICAL LEARNING PROCESSES",
    "CLIMATE CHANGE PRESENTS UNPRECEDENTED GLOBAL CHALLENGES",
    "DNA SEQUENCING REVOLUTIONIZED GENETIC RESEARCH",
    "BLOCKCHAIN TECHNOLOGY ENABLES DECENTRALIZED TRANSACTIONS",
    
    # Literature and philosophy
    "TO BE OR NOT TO BE THAT IS THE QUESTION",
    "THE UNEXAMINED LIFE IS NOT WORTH LIVING",
    "ALL HAPPY FAMILIES ARE ALIKE EACH UNHAPPY FAMILY IS UNHAPPY IN ITS OWN WAY",
    "IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES",
    "IN THE BEGINNING GOD CREATED THE HEAVENS AND THE EARTH",
    
    # Complex structures
    "ALTHOUGH THE THEORY WAS CONTROVERSIAL SCIENTISTS EVENTUALLY ACCEPTED IT",
    "WHENEVER I FEEL AFRAID I HOLD MY HEAD ERECT AND WHISTLE A HAPPY TUNE",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THEN RUNS FAR AWAY",
    "DESPITE THE HEAVY RAIN THE HIKERS CONTINUED THEIR JOURNEY THROUGH THE MOUNTAINS",
    "JUST BECAUSE SOMETHING IS DIFFICULT DOES NOT MEAN IT IS IMPOSSIBLE TO ACHIEVE",
    
    # Quotations
    "BE THE CHANGE YOU WISH TO SEE IN THE WORLD",
    "LIFE IS WHAT HAPPENS WHEN YOU ARE BUSY MAKING OTHER PLANS",
    "THE GREATEST GLORY IN LIVING LIES NOT IN NEVER FALLING BUT IN RISING EVERY TIME WE FALL",
    "THE WAY TO GET STARTED IS TO QUIT TALKING AND BEGIN DOING",
    "YOUR TIME IS LIMITED SO DONT WASTE IT LIVING SOMEONE ELSES LIFE",
    
    # Technical Morse-specific
    "CQ CQ CQ DE AMATEUR RADIO CALLING ANY STATION",
    "QTH IS BOSTON MASSACHUSETTS QSL",
    "SIGNAL REPORT IS FIVE NINE PLUS TWENTY DB",
    "FREQUENCY QSY TO SEVEN POINT ONE ZERO FIVE",
    "QRM HEAVY ON THIS FREQUENCY MOVING UP FIVE",
    
    # Long complex sentences
    "THE INTRICATE RELATIONSHIP BETWEEN QUANTUM PHYSICS AND CONSCIOUSNESS REMAINS ONE OF THE MOST PROFOUND MYSTERIES IN MODERN SCIENCE",
    "NOTWITHSTANDING THE DIFFICULTIES INHERENT IN CROSS CULTURAL COMMUNICATION THE INTERNATIONAL TEAM SUCCESSFULLY COMPLETED THE PROJECT AHEAD OF SCHEDULE",
    "WHEREAS PREVIOUS GENERATIONS RELIED ON TRADITIONAL KNOWLEDGE PASSED DOWN THROUGH APPRENTICESHIP CONTEMPORARY EDUCATION EMPHASIZES CRITICAL THINKING AND INTERDISCIPLINARY APPROACHES",
]

# Domain-specific vocabulary
DOMAIN_VOCABULARIES = {
    "technology": ["ALGORITHM", "BANDWIDTH", "COMPUTER", "DATABASE", "ENCRYPTION", "FIREWALL", 
                  "GIGABYTE", "HARDWARE", "INTERFACE", "JAVASCRIPT", "KERNEL", "LOCALHOST", 
                  "MICROCHIP", "NETWORK", "OPERATING SYSTEM", "PROTOCOL", "QUANTUM", "ROUTER",
                  "SERVER", "TERABYTE", "USB", "VIRTUAL", "WIRELESS", "XML", "YAML", "ZETTABYTE"],
    
    "science": ["ASTRONOMY", "BIOLOGY", "CHEMISTRY", "DENSITY", "ECOSYSTEM", "FUSION", 
               "GRAVITY", "HYPOTHESIS", "ISOTOPE", "JOULE", "KINETIC", "LABORATORY", 
               "MOLECULE", "NEUTRON", "OSMOSIS", "PHOTON", "QUANTUM", "RELATIVITY",
               "SYNTHESIS", "THERMODYNAMICS", "ULTRAVIOLET", "VELOCITY", "WAVELENGTH", 
               "XENON", "YIELD", "ZOOLOGY"],
    
    "morse": ["ANTENNA", "BEACON", "CALLSIGN", "DIT", "DAH", "ELEMENT", "FREQUENCY", "GROUND",
             "HERTZ", "IAMBIC", "KEYER", "LOGBOOK", "MODULATION", "NOISE", "OPERATOR", "PROPAGATION",
             "QRM", "REPEATER", "SIGNAL", "TRANSMITTER", "UTC", "VERTICAL", "WATT", "XRAY", "YANKEE", "ZULU"]
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

def generate_morse_audio(text, noise_level=0.05, speed_factor=1.0, freq_variation=0.0, fading=0.0):
    """
    Generate Morse code audio from text
    
    Args:
        text: Text to convert to Morse code
        noise_level: Amount of noise to add (0 to 1)
        speed_factor: Speed multiplier (higher = faster)
        freq_variation: Amount of frequency variation (0 to 1)
        fading: Amount of signal fading to simulate (0 to 1)
        
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
    
    # Apply fading effect if specified
    if fading > 0:
        try:
            # Get audio as array
            samples = np.array(audio.get_array_of_samples())
            
            # Create fading envelope
            t = np.linspace(0, 1, len(samples))
            fading_points = int(len(samples) * fading * random.uniform(0.1, 0.5))
            fade_positions = sorted(random.sample(range(len(samples) - fading_points), int(len(samples) * fading * 0.1)))
            
            for pos in fade_positions:
                fade_env = np.ones(len(samples))
                fade_env[pos:pos+fading_points] = np.linspace(1, 0.2, fading_points)
                samples = (samples * fade_env).astype(samples.dtype)
                
            # Convert back to AudioSegment
            new_audio = audio._spawn(samples.tobytes())
            audio = new_audio
        except Exception as e:
            print(f"Warning: Could not apply fading effect: {e}")
    
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

def get_nltk_sentences(num_sentences=100):
    """Get sentences from NLTK corpora"""
    sentences = []
    
    try:
        # Try Brown corpus first (more modern English)
        brown_sents = brown.sents()
        if brown_sents:
            for sent in random.sample(list(brown_sents), min(len(brown_sents), num_sentences // 2)):
                # Clean and join the sentence
                text = ' '.join(word.upper() for word in sent if re.match(r'^[A-Za-z0-9]+$', word))
                if len(text) > 10:  # Only keep reasonably long sentences
                    sentences.append(text)
        
        # Add some literary sentences from Gutenberg
        gutenberg_sents = gutenberg.sents()
        if gutenberg_sents:
            for sent in random.sample(list(gutenberg_sents), min(len(gutenberg_sents), num_sentences // 2)):
                text = ' '.join(word.upper() for word in sent if re.match(r'^[A-Za-z0-9]+$', word))
                if len(text) > 10 and len(text) < 100:  # Length constraints
                    sentences.append(text)
    except:
        print("Could not load NLTK corpora. Using built-in sentences only.")
    
    return sentences[:num_sentences]

def generate_grammar_based_sentence():
    """Generate a grammatically structured sentence using templates"""
    templates = [
        "THE {adj} {noun} {verb} {adv}",
        "{adj} {noun} {verb} THE {adj} {noun}",
        "IF {noun} {verb}, THEN {noun} {verb}",
        "{noun} {verb} BECAUSE {noun} {verb}",
        "WHEN {noun} {verb}, {noun} {verb} {adv}",
        "{noun} THAT {verb} {adv} {verb} THE {adj} {noun}",
        "BOTH {noun} AND {noun} {verb} THE {adj} {noun}",
        "NEITHER {noun} NOR {noun} {verb} {adv}",
        "{adv} {verb} THE {noun} BEFORE {noun} {verb}",
        "THE {noun} {verb} WHILE THE {noun} {verb}"
    ]
    
    words = {
        "noun": ["SYSTEM", "NETWORK", "COMPUTER", "DATA", "USER", "SERVER", "CODE", "SIGNAL", 
                "PROGRAM", "STATION", "RADIO", "OPERATOR", "MESSAGE", "FREQUENCY", "ANTENNA",
                "SCIENCE", "THEORY", "EXPERIMENT", "RESEARCHER", "DISCOVERY"],
        "verb": ["TRANSMITS", "RECEIVES", "PROCESSES", "COMPUTES", "ANALYZES", "STORES",
               "BROADCASTS", "DECODES", "ENCRYPTS", "SENDS", "STUDIES", "EXPLORES",
               "EXPLAINS", "VERIFIES", "CONFIRMS", "DEMONSTRATES", "REVEALS"],
        "adj": ["DIGITAL", "ANALOG", "WIRELESS", "ENCRYPTED", "SECURE", "ADVANCED",
              "COMPLEX", "EFFICIENT", "POWERFUL", "RELIABLE", "MODERN", "SCIENTIFIC",
              "THEORETICAL", "EXPERIMENTAL", "QUANTUM", "TECHNICAL", "PRACTICAL"],
        "adv": ["QUICKLY", "SECURELY", "EFFICIENTLY", "WIRELESSLY", "ACCURATELY",
              "RELIABLY", "AUTOMATICALLY", "PRECISELY", "EFFECTIVELY", "CAREFULLY"]
    }
    
    template = random.choice(templates)
    
    # Replace placeholders with random words
    for word_type in words.keys():
        while "{" + word_type + "}" in template:
            template = template.replace("{" + word_type + "}", random.choice(words[word_type]), 1)
            
    return template

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
    
    # Base set of text samples
    base_samples = [
        # Common Morse code phrases
        "HELLO WORLD",
        "SOS",
        "MAYDAY",
        "CQ DX",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "73 BEST REGARDS",
        "QRZ WHO IS CALLING",
        "QTH MY LOCATION IS",
        "QSL CONFIRMING RECEIPT",
        "QRV READY TO RECEIVE",
        
        # Technical terms
        "FREQUENCY SHIFT KEYING",
        "AMPLITUDE MODULATION",
        "CONTINUOUS WAVE",
        "SINGLE SIDEBAND",
        "RADIO FREQUENCY",
        "ELECTROMAGNETIC SPECTRUM",
        "DIGITAL SIGNAL PROCESSING",
        
        # Programming terms
        "PYTHON PROGRAMMING",
        "ARTIFICIAL INTELLIGENCE",
        "MACHINE LEARNING",
        "DEEP NEURAL NETWORKS",
        "CONVOLUTIONAL NETWORKS",
        "RECURRENT NEURAL NETWORKS",
        "NATURAL LANGUAGE PROCESSING",
        
        # Common expressions
        "HOW ARE YOU",
        "THANK YOU",
        "GOOD MORNING",
        "GOOD AFTERNOON",
        "GOOD EVENING",
        "THIS IS A TEST",
        "PLEASE REPLY",
        "OVER AND OUT",
        "ROGER THAT",
    ]
    
    # Extend with advanced sentences
    base_samples.extend(ADVANCED_SENTENCES)
    
    # Add domain-specific vocabulary
    for domain, words in DOMAIN_VOCABULARIES.items():
        for word in words:
            base_samples.append(word)
    
    # Get sentences from NLTK if available (more varied and natural language)
    try:
        nltk_sentences = get_nltk_sentences(200)
        base_samples.extend(nltk_sentences)
    except:
        pass
    
    # Add single letters and numbers
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        base_samples.append(char)
    
    # If custom samples provided, merge them
    if text_samples:
        base_samples.extend(text_samples)
        
    # Ensure uniqueness
    base_samples = list(set(base_samples))
    
    # Create progress bar if verbose
    if verbose:
        pbar = tqdm(total=num_samples, desc="Generating samples")
    
    batch_size = 50  # Save labels every batch_size samples
    
    for i in range(start_index, start_index + num_samples):
        try:
            # Select or generate text
            if i < len(base_samples):
                text = base_samples[i]
            else:
                # Generate more varied text based on various strategies
                text_type = random.choices(
                    ["word", "sentence", "characters", "grammar", "domain", "advanced"],
                    weights=[10, 30, 10, 20, 15, 15]  # Higher weight for sentences
                )[0]
                
                if text_type == "word":
                    # Generate a single random word
                    word_len = random.randint(3, 12)
                    text = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(word_len))
                
                elif text_type == "sentence":
                    # Generate a random sentence
                    word_count = random.randint(3, 12)
                    words = []
                    for _ in range(word_count):
                        word_len = random.randint(2, 10)
                        word = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(word_len))
                        words.append(word)
                    text = ' '.join(words)
                
                elif text_type == "characters":
                    # Generate random characters (useful for training individual letters)
                    chars = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', random.randint(1, 8))
                    text = ' '.join(chars)
                
                elif text_type == "grammar":
                    # Generate grammatically structured sentence
                    text = generate_grammar_based_sentence()
                
                elif text_type == "domain":
                    # Use domain-specific vocabulary
                    domain = random.choice(list(DOMAIN_VOCABULARIES.keys()))
                    words = random.sample(DOMAIN_VOCABULARIES[domain], 
                                         min(random.randint(3, 8), len(DOMAIN_VOCABULARIES[domain])))
                    text = ' '.join(words)
                
                else:  # advanced
                    # Use an advanced sentence or combine parts
                    if random.random() < 0.5 and ADVANCED_SENTENCES:
                        text = random.choice(ADVANCED_SENTENCES)
                    else:
                        # Combine parts of different sentences for more variety
                        parts = []
                        for _ in range(random.randint(2, 3)):
                            if base_samples:
                                sample = random.choice(base_samples).split()
                                if len(sample) > 3:
                                    # Take a random part of the sentence
                                    start = random.randint(0, len(sample) - 3)
                                    end = random.randint(start + 2, min(start + 6, len(sample)))
                                    parts.append(' '.join(sample[start:end]))
                        
                        if parts:
                            text = ' '.join(parts)
                        else:
                            # Fallback to grammar-based
                            text = generate_grammar_based_sentence()
            
            # Ensure the text isn't too long (avoid very long processing times)
            if len(text) > 150:
                text = text[:150]
            
            # Randomize parameters for variety
            speed = random.uniform(0.7, 1.5)  # Wide speed range
            noise = random.uniform(0.01, 0.2)  # More variation in noise
            freq_var = random.uniform(0.0, 0.15)  # Frequency variation
            fading = random.uniform(0.0, 0.3)  # Signal fading simulation
            
            # Generate audio
            audio, elements = generate_morse_audio(
                text, 
                noise_level=noise, 
                speed_factor=speed,
                freq_variation=freq_var,
                fading=fading
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
    parser = argparse.ArgumentParser(description='Generate Morse code audio dataset')
    parser.add_argument('--output', default='./morse_audio_dataset', help='Output directory for dataset')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--start', type=int, default=0, help='Starting index for sample filenames')
    parser.add_argument('--complex', action='store_true', help='Generate more complex and varied sentences')
    
    args = parser.parse_args()
    
    if args.complex:
        print("Generating dataset with enhanced complexity and variety...")
    
    generate_dataset(args.output, num_samples=args.samples, start_index=args.start)