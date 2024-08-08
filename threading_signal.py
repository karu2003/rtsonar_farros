import threading
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import time

class SignalProcessor:
    def __init__(self, Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag):
        self.Qin = Qin
        self.Qdata = Qdata
        self.pulse_a = pulse_a
        self.Nseg = Nseg
        self.Nplot = Nplot
        self.fs = fs
        self.maxdist = maxdist
        self.temperature = temperature
        self.functions = functions
        self.stop_flag = stop_flag

    def process(self):
        while not self.stop_flag.is_set():
            chunk = self.Qin.get()
            print("Received chunk from Qin")  # Отладочное сообщение
            if isinstance(chunk, str) and chunk == "EOT":
                break
            if isinstance(chunk, np.ndarray):
                self.Qdata.put(chunk)
                print("Added chunk to Qdata")  # Отладочное сообщение

def update_plot(Qdata, stop_flag, Nseg):
    plt.ion()  # Включение интерактивного режима
    fig, ax = plt.subplots()
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    line, = ax.plot(np.zeros(Nseg))

    while not stop_flag.is_set():
        if not Qdata.empty():
            print("Qdata is not empty")  # Отладочное сообщение
            chunk = Qdata.get()
            line.set_ydata(chunk)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)
    
    plt.ioff()  # Отключение интерактивного режима
    plt.show()  # Отображение графика после завершения обновления

def generate_data(Qin, stop_flag, Nseg):
    try:
        while not stop_flag.is_set():
            Qin.put(np.random.randn(Nseg))
            print("Added data to Qin")  # Отладочное сообщение
            time.sleep(1)  # Задержка для имитации поступления данных
    except KeyboardInterrupt:
        stop_flag.set()

def main():
    Qin = Queue()
    Qdata = Queue()
    pulse_a = np.array([1, 2, 3])  # Пример данных
    Nseg = 1024
    Nplot = 512
    fs = 44100
    maxdist = 100
    temperature = 25
    functions = []
    stop_flag = threading.Event()

    processor = SignalProcessor(Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag)

    # Запуск потока для обработки сигналов
    signal_thread = threading.Thread(target=processor.process)
    signal_thread.start()

    # Запуск потока для генерации данных
    data_thread = threading.Thread(target=generate_data, args=(Qin, stop_flag, Nseg))
    data_thread.start()

    # Обновление графика в основном потоке
    update_plot(Qdata, stop_flag, Nseg)

    # Ожидание завершения потоков
    signal_thread.join()
    data_thread.join()

if __name__ == "__main__":
    main()