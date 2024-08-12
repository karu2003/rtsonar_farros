import numpy as np
from scipy import signal, interpolate
import threading
import queue
import pyaudio
import time
import matplotlib.pyplot as plt


class RealTimeSonar:
    def __init__(
        self,
        f0: float,
        f1: float,
        fs: float,
        Npulse: int,
        Nseg: int,
        Nrep: int,
        Nplot: int,
        maxdist: float,
        temperature: float,
    ):
        self.f0 = f0
        self.f1 = f1
        self.fs = fs
        self.Npulse = Npulse
        self.Nseg = Nseg
        self.Nrep = Nrep
        self.Nplot = Nplot
        self.maxdist = maxdist
        self.temperature = temperature
        self.Qin, self.Qout, self.Qdata = queue.Queue(), queue.Queue(), queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stop_flag = threading.Event()
        self.chirpPulse = np.zeros(Npulse)

    def genChirpPulse(self):
        Npulse, f0, f1, fs = self.Npulse, self.f0, self.f1, self.fs
        t = np.r_[0 : Npulse / fs : 1 / fs]  # time index
        T = Npulse / fs  # period

        f_of_t = f0 + ((f1 - f0) / (2 * T)) * t  # f of t inside phi of t
        self.chirpPulse = np.exp(1j * 2 * np.pi * f_of_t * t)  # create the chirp signal

    def genPulseTrain(self):
        Nrep, Nseg = self.Nrep, self.Nseg
        chirpPulse = np.real(self.chirpPulse)
        padded_pulse = np.pad(chirpPulse, (0, Nseg - len(chirpPulse)), "constant")
        return np.tile(padded_pulse, Nrep)

    def crossCorr(self, rcv, pulse_a):
        return signal.fftconvolve(rcv, pulse_a[::-1].conj(), mode="full")

    def findDelay(self, Xrcv):
        return np.argmax(Xrcv[: self.Nseg])

    def dist2time(self, dist):
        velocity_of_sound = 331.5 * np.sqrt(1 + self.temperature / 273.15) * 100
        return (dist / velocity_of_sound) * 2

    def put_data(self, ptrain, Twait):
        while not self.stop_flag.is_set():
            if self.Qout.qsize() < 2:
                self.Qout.put(ptrain)
            time.sleep(Twait)
        self.Qout.put(None)

    def play_audio(self, fs, dev=None):
        ostream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=int(fs),
            output=True,
            output_device_index=dev,
        )
        while not self.stop_flag.is_set():
            data = self.Qout.get()
            if data is None:
                break
            ostream.write(data.astype(np.float32).tobytes())
        ostream.stop_stream()
        ostream.close()

    def record_audio(self, fs, dev=None, chunk=2048):
        istream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=int(fs),
            input=True,
            input_device_index=dev,
            frames_per_buffer=chunk,
        )
        while not self.stop_flag.is_set():
            try:
                data_str = istream.read(chunk, exception_on_overflow=False)
                data_flt = np.frombuffer(data_str, dtype="float32")
                self.Qin.put(data_flt)
            except Exception as e:
                print(f"Unexpected error in recording: {e}")
                break
        istream.stop_stream()
        istream.close()
        self.Qin.put(None)

    def signal_process(self, pulse_a):
        Nseg, Nplot, fs, maxdist = (
            self.Nseg,
            self.Nplot,
            self.fs,
            self.maxdist,
        )
        Xrcv = np.zeros(3 * Nseg, dtype="complex")
        cur_idx, found_delay = 0, False
        maxsamp = min(int(self.dist2time(maxdist) * fs), Nseg)

        while not self.stop_flag.is_set():
            chunk = self.Qin.get()
            if chunk is None:
                break

            # self.Qdata.put(chunk)
            Xchunk = self.crossCorr(chunk, pulse_a)
            # self.Qdata.put(np.real(Xchunk[:Nplot]))
            Xchunk = np.reshape(Xchunk, (1, len(Xchunk)))
            try:
                Xrcv[cur_idx : cur_idx + len(chunk) + len(pulse_a) - 1] += Xchunk[0, :]
            except:
                pass
            cur_idx += len(chunk)
            if found_delay and cur_idx >= Nseg:
                if found_delay:
                    idx = self.findDelay(np.abs(Xrcv))
                    Xrcv = np.roll(Xrcv, -idx)
                    Xrcv[-idx:] = 0
                    cur_idx -= idx
                Xrcv_seg = np.sqrt(
                    np.abs(Xrcv[:maxsamp].copy()) / max(np.abs(Xrcv[0]), 1e-5)
                )
                interp = interpolate.interp1d(
                    np.arange(maxsamp),
                    Xrcv_seg,
                    kind="linear",
                    fill_value="extrapolate",
                )
                Xrcv_seg = interp(np.linspace(0, maxsamp - 1, Nplot))
                Xrcv = np.roll(Xrcv, -Nseg)
                Xrcv[-Nseg:] = 0
                cur_idx -= Nseg
                self.Qdata.put(Xrcv_seg)
            elif cur_idx > 2 * Nseg:
                idx = self.findDelay(np.abs(Xrcv))
                Xrcv = np.roll(Xrcv, -idx)
                Xrcv[-idx:] = 0
                cur_idx -= idx + 1
                found_delay = True

        self.Qdata.put(None)

    def start(self):
        self.genChirpPulse()
        hanWin = np.hanning(self.Npulse)
        self.chirpPulse *= hanWin
        ptrain = self.genPulseTrain()

        threads = [
            threading.Thread(
                target=self.put_data, args=(ptrain, self.Nseg / self.fs * 3)
            ),
            threading.Thread(target=self.signal_process, args=(self.chirpPulse,)),
        ]
        for t in threads:
            t.start()

        return threads

    def stop(self):
        self.stop_flag.set()
        self.p.terminate()

    def join_threads(self, threads):
        for t in threads:
            t.join()

    def update_plot(self, Qplot, stop_flag, Nplot):
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        (line,) = ax.plot(np.zeros(Nplot))
        ax.set_ylim(-2, 2)

        while not stop_flag.is_set():
            if not Qplot.empty():
                chunk = Qplot.get()

                if chunk is None:
                    break
                if len(chunk) != Nplot:
                    print(
                        f"Warning: chunk length {len(chunk)} does not match Nplot {Nplot}. Skipping update."
                    )
                    continue
                line.set_ydata(chunk)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.1)

    def start_audio_streams(self):
        play_thread = threading.Thread(target=self.play_audio, args=(self.fs,))
        record_thread = threading.Thread(target=self.record_audio, args=(self.fs,))

        play_thread.start()
        record_thread.start()

        return [play_thread, record_thread]


# Parameters for example usage
fs = 48000
f0 = 6000
f1 = 12000
Npulse = 512  # Length of Chirp Pulse
Nseg = 2048  # Number of samples between pulses (determines maximum time-of-arrival)
Nrep = 24  # Number of repetition in a pulse train (determines vertical length of plot )
Nplot = 200  # Number of pixels in plot along horizontal direction (higher resolution is slower)
maxdist = 200  # Maximum distance in cm
temperature = 20


def main():
    sonar = RealTimeSonar(f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature)

    sonar.stop_flag.clear()

    # Start audio streams from the main thread
    audio_threads = sonar.start_audio_streams()

    # Start processing and data handling in separate threads
    processing_threads = sonar.start()

    # Plot updates must happen in the main thread
    sonar.update_plot(sonar.Qdata, sonar.stop_flag, sonar.Nplot)

    # Wait for processing and plot threads to complete
    sonar.join_threads(processing_threads)

    # Wait for audio threads to complete
    sonar.join_threads(audio_threads)


if __name__ == "__main__":
    main()
