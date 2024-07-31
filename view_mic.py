import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Конфигурация записи
FORMAT = pyaudio.paInt16  # Формат аудио (16-бит PCM)
CHANNELS = 1              # Количество каналов (1 для моно, 2 для стерео)
RATE = 44100              # Частота дискретизации (Hz)
CHUNK = 1024              # Размер блока (количество сэмплов за раз)

# Инициализация PyAudio
p = pyaudio.PyAudio()

# Открываем поток записи
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Инициализация графика
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-', animated=True)
ax.set_xlim(0, CHUNK)
ax.set_ylim(-2**15, 2**15)
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title("Live Audio Signal")

def init():
    """Инициализация функции для FuncAnimation"""
    ax.set_xlim(0, CHUNK)
    ax.set_ylim(-2**15, 2**15)
    return ln,

def update(frame):
    """Функция обновления для FuncAnimation"""
    global xdata, ydata
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    
    # Обновление данных
    xdata.extend(np.arange(CHUNK))
    ydata.extend(data)
    
    # Обрезка старых данных
    xdata = xdata[-CHUNK:]
    ydata = ydata[-CHUNK:]
    
    ln.set_data(np.arange(CHUNK), ydata)
    return ln,

# Запуск анимации
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show()

# Закрытие потока записи
stream.stop_stream()
stream.close()
p.terminate()
