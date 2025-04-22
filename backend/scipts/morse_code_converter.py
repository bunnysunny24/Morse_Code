from pydub import AudioSegment
import os
import datetime
import math
import array
import struct
import tempfile
import subprocess
import sys

MORSE_CODE_DICT = {
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
    ' ': ' ',
    '.': '.-.-.-', ',': '--..--', '?': '..--..',
    '/': '-..-.',  '-': '-....-', '!': '-.-.--'
}

def generate_sine_wave(freq, duration_ms, sample_rate=44100):
    """Generate a sine wave at the given frequency and duration"""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    
    samples = array.array('h')
    amplitude = 32767 
    
    for i in range(num_samples):
        sample = amplitude * math.sin(2 * math.pi * freq * i / sample_rate)
        samples.append(int(sample))
    
    sample_data = struct.pack('<' + ('h' * len(samples)), *samples)
    
    audio = AudioSegment(
        data=sample_data,
        sample_width=2,  
        frame_rate=sample_rate,
        channels=1      
    )
    
    return audio

def play_audio_file(file_path):
    """Play audio using the default system player"""
    try:
        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin': 
            subprocess.call(['open', file_path])
        else:  
            subprocess.call(['xdg-open', file_path])
        print(f"Playing audio file using system player: {file_path}")
    except Exception as e:
        print(f"Could not play the audio file automatically: {e}")
        print(f"Please open and play the file manually from: {file_path}")

DOT_DURATION = 100  
DASH_DURATION = 3 * DOT_DURATION
FREQ = 800 

dot_sound = generate_sine_wave(FREQ, DOT_DURATION)
dash_sound = generate_sine_wave(FREQ, DASH_DURATION)
intra_char_space = AudioSegment.silent(duration=DOT_DURATION)
inter_char_space = AudioSegment.silent(duration=3 * DOT_DURATION)
word_space = AudioSegment.silent(duration=7 * DOT_DURATION)

def text_to_morse_audio(text):
    audio = AudioSegment.silent(duration=0)
    for char in text.upper():
        if char not in MORSE_CODE_DICT:
            continue
        morse = MORSE_CODE_DICT[char]
        if morse == ' ':
            audio += word_space
        else:
            for symbol in morse:
                if symbol == '.':
                    audio += dot_sound
                elif symbol == '-':
                    audio += dash_sound
                audio += intra_char_space
            audio += inter_char_space
    return audio

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)

current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
try:
    current_user = os.getlogin()
except Exception:
    current_user = "Unknown"

print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}")
print(f"Current User's Login: {current_user}")

input_text = input("Enter text to convert to Morse code: ")
if not input_text:
    input_text = "HELLO WORLD"

print(f"Converting '{input_text}' to Morse code...")
morse_audio = text_to_morse_audio(input_text)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = os.path.join(output_dir, f"morse_output_{timestamp}.wav")

try:
    morse_audio.export(output_filename, format="wav")
    print(f"Saved to {os.path.abspath(output_filename)}")
    
    
    play_audio_file(output_filename)
    
except Exception as e:
    print(f"Error saving or playing the file: {e}")
    
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"morse_output_{timestamp}.wav")
        morse_audio.export(temp_filename, format="wav")
        print(f"Saved to temporary location: {temp_filename}")
        play_audio_file(temp_filename)
    except Exception as e2:
        print(f"Failed to save to temporary location as well: {e2}")