import numpy as np
from scipy import signal, interpolate
import asyncio
import queue
import pyaudio
import time
import matplotlib.pyplot as plt

class RealTimeSonar:
    def __init__(self, f0: float, f1: float, fs: float, Npulse: int, Nseg: int, Nrep: int, Nplot: int, maxdist: float, temperature: float):
        self.f0 = f0
        self.f1 = f1
        self.fs = fs
        self.Npulse = Npulse
        self.Nseg = Nseg
        self.Nrep = Nrep
        self.Nplot = Nplot
        self.maxdist = maxdist
        self.temperature = temperature
        self.Qout, self.Qdata = queue.Queue(), queue.Queue()
        self.stop_flag = asyncio.Event()
        self.p = pyaudio.PyAudio()

    def genChirpPulse(self):
        Npulse, f0, f1, fs = self.Npulse, self.f0, self.f1, self.fs
        t = np.linspace(0, Npulse / fs, Npulse, endpoint=False)
        return np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) / (2 * Npulse / fs) * t**2))

    def genPulseTrain(self, pulse):
        Nrep, Nseg = self.Nrep, self.Nseg
        padded_pulse = np.pad(pulse, (0, Nseg - len(pulse)), "constant")
        return np.tile(padded_pulse, Nrep)

    def crossCorr(self, rcv, pulse_a):
        return signal.fftconvolve(rcv, pulse_a[::-1].conj(), mode="full")

    def findDelay(self, Xrcv):
        return np.argmax(Xrcv[:self.Nseg])

    def dist2time(self, dist):
        velocity_of_sound = 331.5 * np.sqrt(1 + self.temperature / 273.15) * 100
        return (dist / velocity_of_sound) * 2

    async def put_data(self, ptrain, Twait):
        while not self.stop_flag.is_set():
            if self.Qout.qsize() < 2:
                self.Qout.put(ptrain)
            await asyncio.sleep(Twait)
        self.Qout.put(None)

    async def play_audio(self, fs, dev=None):
        ostream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs), output=True, output_device_index=dev)
        while not self.stop_flag.is_set():
            data = self.Qout.get()
            if data is None:
                break
            ostream.write(data.astype(np.float32).tobytes())
        ostream.stop_stream()
        ostream.close()

    async def record_audio(self, fs, dev=None, chunk=2048):
        istream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs), input=True, input_device_index=dev, frames_per_buffer=chunk)
        while not self.stop_flag.is_set():
            try:
                data_str = istream.read(chunk, exception_on_overflow=False)
                data_flt = np.frombuffer(data_str, dtype="float32")
                self.Qdata.put(data_flt)
            except Exception as e:
                print(f"Unexpected error in recording: {e}")
                break
        istream.stop_stream()
        istream.close()
        self.Qdata.put(None)

    async def signal_process(self, pulse_a):
        Nseg, Nplot, fs, maxdist = self.Nseg, self.Nplot, self.fs, self.maxdist
        Xrcv = np.zeros(3 * Nseg, dtype="complex")
        cur_idx, found_delay = 0, False
        maxsamp = min(int(self.dist2time(maxdist) * fs), Nseg)

        while not self.stop_flag.is_set():
            chunk = await asyncio.to_thread(self.Qdata.get)
            if chunk is None:
                break
            Xchunk = self.crossCorr(chunk, pulse_a)
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
                Xrcv_seg = np.sqrt(np.abs(Xrcv[:maxsamp].copy()) / max(np.abs(Xrcv[0]), 1e-5))
                interp = interpolate.interp1d(np.arange(maxsamp), Xrcv_seg, kind="linear", fill_value="extrapolate")
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

    async def update_plot(self, Nplot):
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        line, = ax.plot(np.zeros(Nplot))

        while not self.stop_flag.is_set():
            if not self.Qdata.empty():
                chunk = self.Qdata.get()
                if chunk is None:
                    break
                if len(chunk) != Nplot:
                    print(f"Warning: chunk length {len(chunk)} does not match Nplot {Nplot}. Skipping update.")
                    continue
                line.set_ydata(chunk)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.1)
        
        plt.ioff()  # Disable interactive mode
        plt.show()  # Show plot after updating

    async def start(self):
        pulse_a = self.genChirpPulse()
        hanWin = np.hanning(self.Npulse)
        pulse_a *= hanWin
        ptrain = self.genPulseTrain(np.real(pulse_a))
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.put_data(ptrain, self.Nseg / self.fs * 3)),
            asyncio.create_task(self.record_audio(self.fs)),
            asyncio.create_task(self.play_audio(self.fs)),
            asyncio.create_task(self.signal_process(pulse_a)),
            asyncio.create_task(self.update_plot(self.Nplot)),
        ]
        
        await asyncio.gather(*tasks)

    def stop(self):
        self.stop_flag.set()
        self.p.terminate()

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

async def main():
    sonar = RealTimeSonar(f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature)
    try:
        await sonar.start()
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
    finally:
        sonar.stop()

if __name__ == "__main__":
    asyncio.run(main())
