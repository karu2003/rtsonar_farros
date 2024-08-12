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

        # Блокировка для синхронизации передачи и приёма
        self.tx_lock = threading.Lock()

        # Порог корреляции для проверки
        self.correlation_threshold = 0.5

        # Размер буфера приёма
        self.chunk_size = 2048
        self.receive_buffer = np.zeros(self.chunk_size * 2)  # Длина буфера в два раза больше размера блока

        # Очереди временных меток
        self.send_time_queue = queue.Queue()
        self.receive_time_queue = queue.Queue()

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
        )

        while not self.stop_flag.is_set():
            with self.tx_lock:
                # Запоминаем время отправки
                send_time = time.time()
                self.send_time_queue.put(send_time)

                stream.write(self.chirp_signal.astype(np.float32).tobytes())

            # Ожидание, не блокируя другие потоки
            self.pause_event.wait(self.pause_duration)

        stream.stop_stream()
        stream.close()

    def record_signal(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=int(self.fs),
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        while not self.stop_flag.is_set():
            # Приостанавливаем запись на время передачи
            with self.tx_lock:
                # Время получения данных
                receive_time = time.time()
                self.receive_time_queue.put(receive_time)

                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.record_queue.put(audio_chunk)

        stream.stop_stream()
        stream.close()

    def process_signals(self):
        while not self.stop_flag.is_set() or not self.record_queue.empty():
            try:
                chunk = self.record_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Обновляем буфер приёма, добавляя новые данные в конец и удаляя старые
            self.receive_buffer = np.roll(self.receive_buffer, -self.chunk_size)
            self.receive_buffer[-self.chunk_size:] = chunk

            # Вычисляем корреляцию в окне
            correlation = fftconvolve(
                self.receive_buffer, self.chirp_signal[::-1], mode="valid"
            )
            peak_value = np.max(np.abs(correlation))
            delay_index = np.argmax(np.abs(correlation))
            signal_delay = delay_index / self.fs

            # Проверяем наличие значимого сигнала
            if peak_value > self.correlation_threshold:
                if not self.send_time_queue.empty() and not self.receive_time_queue.empty():
                    send_time = self.send_time_queue.get()
                    receive_time = self.receive_time_queue.get()

                    # Проверка временных меток на корректность
                    if (receive_time - send_time) > 0:
                        # Общая задержка сигнала
                        total_delay = receive_time - send_time + signal_delay

                        # Проверяем, чтобы задержка не была меньше минимального расстояния
                        if total_delay >= (self.min_distance / self.velocity_of_sound):
                            distance = (total_delay * self.velocity_of_sound) / 2
                            print(f"Вычисленная дистанция: {distance:.4f} метров")
                        else:
                            print("Игнорирование данных: разница между временем передачи и приемом меньше минимального расстояния.")
                    else:
                        print("Игнорирование данных: некорректное время приема.")
                else:
                    print("Игнорирование данных: отсутствуют временные метки для расчета дистанции.")
            else:
                print("Игнорирование данных: корреляция ниже порога.")

            # Отправка данных в очередь для визуализации
            self.plot_queue.put(self.receive_buffer.copy())

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
max_distance = 20  # Максимальное измеряемое расстояние (метры)
min_distance = 1  # Минимальное измеряемое расстояние (метры)
salinity = 35  # Соленость воды (только если medium="water")
medium = "air"  # Среда передачи звука ("air" или "water")

# Создаем объект RealTimeSonar
sonar = RealTimeSonar(f0, f1, fs, temperature, max_distance, min_distance, salinity, medium)

# Запуск работы программы
sig.signal(sig.SIGINT, signal_handler)

play_thread, record_thread, process_thread = sonar.start()

print("Программа работает. Нажмите Ctrl+C для остановки...")  # Сообщение о работе программы

try:
    sonar.update_plot()
except KeyboardInterrupt:
    signal_handler(sig.SIGINT, None)
