import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import queue
import threading

# Configuration for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream for recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Queue to store audio data
data_queue = queue.Queue()

# Flag to stop recording
stop_flag = threading.Event()

def audio_callback(in_data, frame_count, time_info, status):
    if not stop_flag.is_set():
        data_queue.put(in_data)
    return (None, pyaudio.paContinue)

# Start audio stream
stream.start_stream()

# Initialize plot
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-', animated=True)
ax.set_xlim(0, CHUNK)
ax.set_ylim(-2**15, 2**15)
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title("Live Audio Signal")

def init():
    """Initialization function for FuncAnimation"""
    ax.set_xlim(0, CHUNK)
    ax.set_ylim(-2**15, 2**15)
    return ln,

def update(frame):
    """Update function for FuncAnimation"""
    if not data_queue.empty():
        data = data_queue.get()
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Update data
        xdata.extend(np.arange(CHUNK))
        ydata.extend(audio_data)
        
        # Trim old data
        xdata = xdata[-CHUNK:]
        ydata = ydata[-CHUNK:]
        
        ln.set_data(np.arange(CHUNK), ydata)
    return ln,

def on_key_press(event):
    """Function to handle key press events"""
    if event.char == 'q':
        stop_flag.set()

# Create GUI for keypress handling
root = tk.Tk()
root.withdraw()
root.bind("<KeyPress>", on_key_press)

# Start animation
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show(block=False)
root.mainloop()

# Stop and close stream
stream.stop_stream()
stream.close()
p.terminate()
