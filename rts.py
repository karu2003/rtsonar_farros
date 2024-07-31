import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from scipy import signal, interpolate
import threading
import queue
import pyaudio
import time
from threading import Event
from typing import Callable, List


def put_data(Qout: queue.Queue, ptrain: np.ndarray, Twait: float, stop_flag: Event):
    while not stop_flag.is_set():
        if Qout.qsize() < 2:
            Qout.put(ptrain)
        time.sleep(Twait)
    Qout.put(None)  # Use None instead of "EOT"


def play_audio(
    Qout: queue.Queue, p: pyaudio.PyAudio, fs: float, stop_flag: Event, dev: int = None
):
    ostream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=int(fs),
        output=True,
        output_device_index=dev,
    )
    while not stop_flag.is_set():
        data = Qout.get()
        if data is None:
            break
        ostream.write(data.astype(np.float32).tobytes())
    ostream.stop_stream()
    ostream.close()


def record_audio(
    Qin: queue.Queue,
    p: pyaudio.PyAudio,
    fs: float,
    stop_flag: Event,
    dev: int = None,
    chunk: int = 2048,
):
    istream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=int(fs),
        input=True,
        input_device_index=dev,
        frames_per_buffer=chunk,
    )
    while not stop_flag.is_set():
        try:
            data_str = istream.read(chunk, exception_on_overflow=False)
            data_flt = np.frombuffer(data_str, dtype="float32")
            Qin.put(data_flt)
        except Exception as e:
            print(f"Unexpected error in recording: {e}")
            break
    istream.stop_stream()
    istream.close()
    Qin.put(None)  # Use None instead of "EOT"


def signal_process(
    Qin: queue.Queue,
    Qdata: queue.Queue,
    pulse_a: np.ndarray,
    Nseg: int,
    Nplot: int,
    fs: float,
    maxdist: float,
    temperature: float,
    functions: List[Callable],
    stop_flag: Event,
):
    crossCorr, findDelay, dist2time = functions[2], functions[3], functions[4]
    Xrcv = np.zeros(3 * Nseg, dtype="complex")
    cur_idx, found_delay = 0, False
    maxsamp = min(int(dist2time(maxdist, temperature) * fs), Nseg)

    while not stop_flag.is_set():
        chunk = Qin.get()
        if chunk is None:
            break
        Xchunk = crossCorr(chunk, pulse_a)
        Xchunk = np.reshape(Xchunk, (1, len(Xchunk)))

        try:
            Xrcv[cur_idx : cur_idx + len(chunk) + len(pulse_a) - 1] += Xchunk[0, :]
        except:
            pass

        cur_idx += len(chunk)
        if found_delay and cur_idx >= Nseg:
            if found_delay:
                idx = findDelay(np.abs(Xrcv), Nseg)
                Xrcv = np.roll(Xrcv, -idx)
                Xrcv[-idx:] = 0
                cur_idx -= idx
            Xrcv_seg = np.sqrt(
                np.abs(Xrcv[:maxsamp].copy()) / max(np.abs(Xrcv[0]), 1e-5)
            )
            interp = interpolate.interp1d(
                np.arange(maxsamp), Xrcv_seg, kind="linear", fill_value="extrapolate"
            )
            Xrcv_seg = interp(np.linspace(0, maxsamp - 1, Nplot))
            Xrcv = np.roll(Xrcv, -Nseg)
            Xrcv[-Nseg:] = 0
            cur_idx -= Nseg
            Qdata.put(Xrcv_seg)
        elif cur_idx > 2 * Nseg:
            idx = findDelay(np.abs(Xrcv), Nseg)
            Xrcv = np.roll(Xrcv, -idx)
            Xrcv[-idx:] = 0
            cur_idx -= idx + 1
            found_delay = True

    Qdata.put(None)  # Use None instead of "EOT"


def stop_on_keypress(stop_flag: Event):
    input("Press any key to stop the sonar...\n")
    stop_flag.set()


def rtsonar(
    f0: float,
    f1: float,
    fs: float,
    Npulse: int,
    Nseg: int,
    Nrep: int,
    Nplot: int,
    maxdist: float,
    temperature: float,
    functions: List[Callable],
):
    genChirpPulse, genPulseTrain = functions[0], functions[1]
    pulse_a = genChirpPulse(Npulse, f0, f1, fs)
    hanWin = np.hanning(Npulse)
    pulse_a *= hanWin
    ptrain = genPulseTrain(np.real(pulse_a), Nrep, Nseg)
    Qin, Qout, Qdata = queue.Queue(), queue.Queue(), queue.Queue()
    p = pyaudio.PyAudio()
    global img  # Declare img as global so that it can be accessed and modified in update
    img = np.zeros((Nrep, Nplot, 3), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Sonar")
    ax.set_xlabel("Distance [cm]")
    ax.set_ylabel("Time [s]")

    # Initialize the imshow object for the animation
    im = ax.imshow(img, aspect="auto", extent=[0, maxdist, 0, Nrep * Nseg / fs])

    def init():
        im.set_array(img)
        return [im]

    def update(frame):
        global img  # Declare img as global to modify it
        new_line = Qdata.get()
        if new_line is not None:
            new_line = np.minimum(
                new_line / max(np.percentile(new_line, 97), 1e-5), 1
            ) ** (1 / 1.8)
            img = np.roll(img, 1, axis=0)
            img[0, :] = cm.jet(new_line)[:, :3]
            im.set_array(img)
        return [im]

    stop_flag = threading.Event()
    threads = [
        threading.Thread(
            target=put_data, args=(Qout, ptrain, Nseg / fs * 3, stop_flag)
        ),
        threading.Thread(target=record_audio, args=(Qin, p, fs, stop_flag)),
        threading.Thread(target=play_audio, args=(Qout, p, fs, stop_flag)),
        threading.Thread(
            target=signal_process,
            args=(
                Qin,
                Qdata,
                pulse_a,
                Nseg,
                Nplot,
                fs,
                maxdist,
                temperature,
                functions,
                stop_flag,
            ),
        ),
    ]
    for t in threads:
        t.start()

    stop_thread = threading.Thread(target=stop_on_keypress, args=(stop_flag,))
    stop_thread.start()

    ani = FuncAnimation(
        fig, update, init_func=init, blit=True, interval=50, cache_frame_data=False
    )
    plt.show()

    stop_thread.join()

    for t in threads:
        t.join()

    p.terminate()


# Example function definitions (replace with actual implementations)
def genChirpPulse(Npulse, f0, f1, fs):
    t = np.linspace(0, Npulse / fs, Npulse, endpoint=False)
    return np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) / (2 * Npulse / fs) * t**2))


def genPulseTrain(pulse, Nrep, Nseg):
    padded_pulse = np.pad(pulse, (0, Nseg - len(pulse)), "constant")
    return np.tile(padded_pulse, Nrep)


def crossCorr(rcv, pulse_a):
    return signal.fftconvolve(rcv, pulse_a[::-1].conj(), mode="full")


def findDelay(Xrcv, Nseg):
    return np.argmax(Xrcv[:Nseg])


def dist2time(dist, temperature=20):
    velocity_of_sound = 331.5 * np.sqrt(1 + temperature / 273.15) * 100
    return (dist / velocity_of_sound) * 2


# Parameters for example usage
fs = 48000
f0 = 6000
f1 = 16000
Npulse = 1024
Nseg = 2048
Nrep = 30
Nplot = 256
maxdist = 100
temperature = 20
functions = [genChirpPulse, genPulseTrain, crossCorr, findDelay, dist2time]

rtsonar(f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature, functions)
