import numpy as np
from scipy.signal import fftconvolve
import pyaudio
import threading
import queue
import matplotlib.pyplot as plt
import signal as sig
import sys
import time


class RealTimeSonar:
    def __init__(
        self,
        f0,
        f1,
        fs,
        temperature,
        max_distance,
        min_distance,
        salinity=35,
        medium="air",
    ):
        self.f0 = f0
        self.f1 = f1
        self.fs = fs
        self.temperature = temperature
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.salinity = salinity
        self.medium = medium

        # Определяем скорость звука в выбранной среде
        if medium == "air":
            self.velocity_of_sound = 331.5 * np.sqrt(1 + self.temperature / 273.15)
        elif medium == "water":
            self.velocity_of_sound = self.calculate_velocity_of_sound_water(
                self.temperature, self.salinity
            )
        else:
            raise ValueError("Среда должна быть 'air' или 'water'")

        # Вычисляем максимальную длительность сигнала, исходя из минимального расстояния
        self.max_duration = self.calculate_max_duration_for_distance(min_distance)

        # Устанавливаем длительность сигнала в зависимости от максимальной длительности
        self.duration = self.max_duration
        self.chirp_signal = self.generate_chirp()

        self.record_queue = queue.Queue()  # Очередь для записи аудио
        self.plot_queue = queue.Queue()  # Очередь для визуализации
        self.stop_flag = threading.Event()

        self.p = pyaudio.PyAudio()

        # Расчет времени паузы между сигналами
        self.pause_duration = self.calculate_pause_duration()

        # Событие для реализации паузы
        self.pause_event = threading.Event()

    def generate_chirp(self):
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        chirp_signal = np.sin(
            2 * np.pi * (self.f0 * t + (self.f1 - self.f0) / (2 * self.duration) * t**2)
        )
        return chirp_signal

    def calculate_velocity_of_sound_water(self, temperature, salinity):
        # Формула для расчета скорости звука в воде с учетом температуры и солёности
        return (
            1449.2
            + 4.6 * temperature
            - 0.055 * temperature**2
            + 0.00029 * temperature**3
            + (1.34 - 0.01 * temperature) * (salinity - 35)
        )

    def calculate_pause_duration(self):
        # Время для прохождения сигнала до максимального расстояния и обратно
        round_trip_time = (2 * self.max_distance) / self.velocity_of_sound
        return round_trip_time

    def calculate_max_duration_for_distance(self, distance):
        # Время для прохождения сигнала до указанного расстояния
        return distance / self.velocity_of_sound

    def play_chirp(self):
        stream = self.p.open(
            format=pyaudio.paFloat32, channels=1, rate=int(self.fs), output=True
        )  # , frames_per_buffer=2048)

        while not self.stop_flag.is_set():
            stream.write(self.chirp_signal.astype(np.float32).tobytes())

            # Ожидание, не блокируя другие потоки
            self.pause_event.wait(self.pause_duration)

        stream.stop_stream()
        stream.close()

    def record_signal(self):
        chunk = 2048
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=int(self.fs),
            input=True,
            frames_per_buffer=chunk,
        )

        while not self.stop_flag.is_set():
            data = stream.read(chunk, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            self.record_queue.put(audio_chunk)

        stream.stop_stream()
        stream.close()

    def process_signals(self):
        chirp_length = len(self.chirp_signal)
        recorded_chunks = []

        while not self.stop_flag.is_set() or not self.record_queue.empty():
            try:
                chunk = self.record_queue.get(
                    timeout=1
                )  # Timeout to periodically check stop_flag
            except queue.Empty:
                continue

            if chunk is not None:
                recorded_chunks.append(chunk)
                # Если накоплено достаточно данных
                if len(recorded_chunks) * len(chunk) >= chirp_length:
                    recorded_signal = np.concatenate(recorded_chunks)
                    correlation = fftconvolve(
                        recorded_signal, self.chirp_signal[::-1], mode="full"
                    )
                    delay_index = np.argmax(np.abs(correlation))
                    delay = delay_index / self.fs

                    distance = (delay * self.velocity_of_sound) / 2

                    # Вывод дистанции в консоль
                    print(f"Вычисленная дистанция: {distance:.4f} метров")

                    # Отправка данных в очередь для визуализации
                    self.plot_queue.put(recorded_signal)

                    # Сброс списка записанных данных для следующего расчета
                    recorded_chunks = []

    def update_plot(self):
        plt.ion()  # Включаем интерактивный режим
        fig, ax = plt.subplots()
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")

        # Установим пустой график
        num_samples = int(self.fs * self.duration)
        (line,) = ax.plot(np.zeros(num_samples))

        while not self.stop_flag.is_set():
            if not self.plot_queue.empty():
                # Извлечение данных из очереди для визуализации
                signal_to_plot = self.plot_queue.get()

                if len(signal_to_plot) == num_samples:
                    line.set_ydata(signal_to_plot)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()

                plt.pause(0.1)
            else:
                # Если данных нет, делаем паузу
                plt.pause(0.1)

        plt.ioff()  # Выключаем интерактивный режим
        plt.show()

    def start(self):
        play_thread = threading.Thread(target=self.play_chirp)
        record_thread = threading.Thread(target=self.record_signal)
        process_thread = threading.Thread(target=self.process_signals)

        play_thread.start()
        record_thread.start()
        process_thread.start()

        return play_thread, record_thread, process_thread

    def stop(self):
        self.stop_flag.set()

        # Ожидание завершения потоков
        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                thread.join()

        self.p.terminate()


def signal_handler(sig, frame):
    print("\nОстановка программы...")
    sonar.stop()
    sys.exit(0)


# Параметры
fs = 48000  # Частота дискретизации
f0 = 6000  # Начальная частота чирпа (Гц)
f1 = 12000  # Конечная частота чирпа (Гц)
temperature = 20  # Температура в градусах Цельсия
max_distance = 20  # Максимальное измеряемое расстояние (м)
min_distance = 0.5  # Минимальное измеряемое расстояние (м)
salinity = 35  # Солёность воды в промилле
medium = "air"  # Среда (может быть 'air' или 'water')

if __name__ == "__main__":
    # Регистрация обработчика сигнала для прерывания Ctrl+C
    sig.signal(sig.SIGINT, signal_handler)

    try:
        sonar = RealTimeSonar(
            f0, f1, fs, temperature, max_distance, min_distance, salinity, medium
        )
    except ValueError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

    # plt.figure()
    # plt.plot(sonar.chirp_signal)
    # plt.title("Chirp Signal")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")
    # plt.show()
    # exit()

    # Запуск потоков
    play_thread, record_thread, process_thread = sonar.start()

    # Обновление графика в главном потоке
    sonar.update_plot()

    # Ожидание завершения работы
    print("Программа работает. Нажмите Ctrl+C для остановки...")

    # Остановка потоков и завершение работы
    sonar.stop()
