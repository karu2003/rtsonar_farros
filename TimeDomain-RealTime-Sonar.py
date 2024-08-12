import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from numpy import r_, pi, exp
import threading, time, queue, pyaudio
from matplotlib.cm import viridis

# Define constants
CHUNK_SIZE = 2048
DURATION = 30  # Duration for sonar operation in seconds


def put_data(output_queue, pulse_train, wait_time, stop_event):
    """
    Generates and puts pulse train data into the output queue periodically.
    """
    while not stop_event.is_set():
        if output_queue.qsize() < 2:
            output_queue.put(pulse_train)
        time.sleep(wait_time)
    output_queue.put("EOT")


def play_audio(output_queue, pyaudio_instance, sample_rate, stop_event, device=None):
    """
    Plays audio from the output queue using PyAudio.
    """
    stream = pyaudio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=int(sample_rate),
        output=True,
        output_device_index=device,
    )

    while not stop_event.is_set():
        data = output_queue.get()

        # Ensure we handle the end-of-transmission signal
        if isinstance(data, str) and data == "EOT":
            break

        if isinstance(data, np.ndarray):
            try:
                stream.write(data.astype(np.float32).tobytes())
            except IOError:
                break

    stream.stop_stream()
    stream.close()



def record_audio(
    input_queue,
    pyaudio_instance,
    sample_rate,
    stop_event,
    device=None,
    chunk_size=CHUNK_SIZE,
):
    """
    Records audio and puts it into the input queue in chunks.
    """
    stream = pyaudio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=int(sample_rate),
        input=True,
        input_device_index=device,
        frames_per_buffer=chunk_size,
    )

    while not stop_event.is_set():
        try:
            data_str = stream.read(chunk_size, exception_on_overflow=False)
            data_flt = np.frombuffer(data_str, dtype="float32")
            input_queue.put(data_flt)
        except IOError:
            break

    stream.stop_stream()
    stream.close()
    input_queue.put("EOT")


def signal_process(
    input_queue, data_queue, pulse_a, segment_length, num_plot, sample_rate, max_distance, temperature, functions, stop_event
):
    """
    Processes incoming audio data to extract and process sonar signals.
    """
    cross_correlation = functions['cross_correlation']
    find_delay = functions['find_delay']
    distance_to_time = functions['distance_to_time']

    # Initialize x_received to accommodate 3 segments worth of data
    x_received = np.zeros(3 * segment_length, dtype="complex")
    current_index = 0
    delay_found = False
    max_samples = min(int(distance_to_time(max_distance, temperature) * sample_rate), segment_length)

    while not stop_event.is_set():
        chunk = input_queue.get()
        
        # Ensure we handle the end-of-transmission signal
        if isinstance(chunk, str) and chunk == "EOT":
            break
        
        if isinstance(chunk, np.ndarray):
            # Perform cross-correlation
            x_chunk = cross_correlation(chunk, pulse_a)

            # Ensure x_chunk is reshaped to be a flat array
            x_chunk = np.reshape(x_chunk, (1, len(x_chunk)))

            # Determine how many elements to add to x_received without exceeding its bounds
            end_index = min(current_index + len(chunk) + len(pulse_a) - 1, len(x_received))
            chunk_size = end_index - current_index

            # Add only the part of x_chunk that fits into x_received
            x_received[current_index:end_index] += x_chunk[0, :chunk_size]

            # Update the current index
            current_index += len(chunk)

            if delay_found and (current_index >= segment_length):
                idx = find_delay(abs(x_received), segment_length)
                x_received = np.roll(x_received, -idx)
                x_received[-idx:] = 0
                current_index -= idx

                x_received_segment = (abs(x_received[:max_samples].copy()) / np.maximum(abs(x_received[0]), 1e-5)) ** 0.5
                interp = interpolate.interp1d(r_[:max_samples], x_received_segment)
                x_received_segment = interp(r_[: max_samples - 1 : (num_plot * 1j)])

                x_received = np.roll(x_received, -segment_length)
                x_received[-segment_length:] = 0
                current_index -= segment_length

                data_queue.put(x_received_segment)

            elif current_index > 2 * segment_length:
                idx = find_delay(abs(x_received), segment_length)
                x_received = np.roll(x_received, -idx)
                x_received[-idx:] = 0
                current_index -= idx + 1
                delay_found = True

    data_queue.put("EOT")



def image_update(data_queue, img, num_repeats, num_plot, stop_event):
    """
    Updates the image with new data for real-time plotting.
    """
    while not stop_event.is_set():
        try:
            new_line = data_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Ensure we handle the end-of-transmission signal
        if isinstance(new_line, str) and new_line == "EOT":
            print("End of transmission")
            break

        if isinstance(new_line, np.ndarray):
            # Normalize and transform the new line for display
            new_line = np.minimum(new_line / np.maximum(np.percentile(new_line, 97), 1e-5), 1) ** (1 / 1.8)

            # Update the image buffer
            img = np.roll(img, 1, 0)
            view = img.view(dtype=np.uint8).reshape((num_repeats, num_plot, 4))
            view[0, :, :] = (viridis(new_line) * 255)

            # Display the updated image
            plt.imshow(img)
            plt.pause(0.01)



def rtsonar(
    f0,
    f1,
    fs,
    pulse_length,
    segment_length,
    num_repeats,
    num_plot,
    max_distance,
    temperature,
    functions,
):
    """
    Main function to run real-time sonar.
    """
    generate_chirp_pulse = functions["generate_chirp_pulse"]
    generate_pulse_train = functions["generate_pulse_train"]

    pulse_a = generate_chirp_pulse(pulse_length, f0, f1, fs)
    pulse_a *= np.hanning(pulse_length)
    pulse = np.real(pulse_a)
    pulse_train = generate_pulse_train(pulse, num_repeats, segment_length)

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    data_queue = queue.Queue()

    pyaudio_instance = pyaudio.PyAudio()

    img = np.zeros((num_repeats, num_plot), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((num_repeats, num_plot, 4))
    view[:, :, 3] = 255

    stop_event = threading.Event()

    threads = [
        threading.Thread(
            target=put_data,
            args=(output_queue, pulse_train, segment_length / fs, stop_event),
        ),
        threading.Thread(
            target=record_audio, args=(input_queue, pyaudio_instance, fs, stop_event)
        ),
        threading.Thread(
            target=play_audio, args=(output_queue, pyaudio_instance, fs, stop_event)
        ),
        threading.Thread(
            target=signal_process,
            args=(
                input_queue,
                data_queue,
                pulse_a,
                segment_length,
                num_plot,
                fs,
                max_distance,
                temperature,
                functions,
                stop_event,
            ),
        ),
        threading.Thread(
            target=image_update,
            args=(data_queue, img, num_repeats, num_plot, stop_event),
        ),
    ]

    for thread in threads:
        thread.start()

    time_end = time.time() + DURATION
    while time.time() < time_end:
        time.sleep(1)

    stop_event.set()

    for thread in threads:
        thread.join()

    pyaudio_instance.terminate()
    print("All threads joined and PyAudio terminated.")


def generate_chirp_pulse(pulse_length, f0, f1, sample_rate):
    """
    Generates a chirp pulse.
    """
    t = r_[0:pulse_length] / sample_rate
    return signal.chirp(t, f0, t[-1], f1) * exp(1j * 2 * pi * f0 * t)


def generate_pulse_train(pulse, num_repeats, segment_length):
    """
    Generates a train of pulses.
    """
    pulse_train = np.zeros((segment_length * num_repeats))
    for idx in range(num_repeats):
        start = idx * segment_length
        pulse_train[start : start + len(pulse)] = pulse
    return pulse_train


def cross_correlation(data, pulse):
    """
    Calculates cross-correlation using FFT.
    """
    return signal.fftconvolve(data, pulse[::-1], mode="full")


def find_delay(correlated_data, segment_length):
    """
    Finds the delay using the correlated data.
    """
    return np.argmax(abs(correlated_data[:segment_length]))


def distance_to_time(max_distance, temperature):
    """
    Calculates the time for a round trip based on the maximum distance and temperature.
    """
    c = 331.5 * np.sqrt(1 + (temperature / 273.15))
    max_distance_m = max_distance / 100
    return (2 * max_distance_m) / c


# Function dictionary
functions = {
    "generate_chirp_pulse": generate_chirp_pulse,
    "generate_pulse_train": generate_pulse_train,
    "cross_correlation": cross_correlation,
    "find_delay": find_delay,
    "distance_to_time": distance_to_time,
}

# Real-time sonar parameters
f0 = 6000  # start frequency of chirp
f1 = 12000  # stop frequency of chirp
fs = 48000  # sampling frequency
pulse_length = 512  # number of samples in chirp
segment_length = 2048  # number of samples between chirps
num_repeats = 23  # number of chirps to display
num_plot = 200  # number of samples to plot
max_distance = 200  # maximum distance in cm
temperature = 22  # temperature in Celsius

# Run sonar
rtsonar(
    f0,
    f1,
    fs,
    pulse_length,
    segment_length,
    num_repeats,
    num_plot,
    max_distance,
    temperature,
    functions,
)
