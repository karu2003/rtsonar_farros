import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from rtsonar import rtsonar
import warnings
import threading

warnings.simplefilter(action='ignore', category=FutureWarning)

# Генерация Чирпового импульса
def genChirpPulse(Npulse, f0, f1, fs):
    t = np.linspace(0, Npulse/fs, Npulse, endpoint=False)
    chirpPulse = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) / (2 * Npulse/fs) * t**2))
    return chirpPulse

# Генерация последовательности импульсов
def genPulseTrain(pulse, Nrep, Nseg):
    padded_pulse = np.pad(pulse, (0, Nseg - len(pulse)), 'constant')
    ptrain = np.tile(padded_pulse, Nrep)
    return ptrain

# Кросс-корреляция
def crossCorr(rcv, pulse_a):
    Xrcv = signal.fftconvolve(rcv, pulse_a[::-1].conj(), mode="full")
    return Xrcv

# Нахождение задержки
def findDelay(Xrcv, Nseg):
    return np.argmax(Xrcv[:Nseg])

# Перевод расстояния в время
def dist2time(dist, temperature=21):
    velocity_of_sound = 331.5 * np.sqrt(1 + temperature / 273.15) * 100
    time = (dist / velocity_of_sound) * 2
    return time

# Функция для остановки сонара по нажатию клавиши
def stop_on_keypress(stop_flag):
    input("Нажмите любую клавишу для остановки сонара...\n")
    stop_flag.set()

# Основная часть программы для работы с реальным временем
fs = 48000
f0 = 6000
f1 = 12000
DurationS = 0.00001
Pause = 0.1
Npulse = 500
Nrep = 24
Nseg = int(fs * Pause)
Nplot = 1024
maxdist = 200
temperature = 20

functions = (genChirpPulse, genPulseTrain, crossCorr, findDelay, dist2time)

# Запуск сонара
stop_flag = rtsonar(f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature, functions)

# Запуск потока для остановки сонара по нажатию клавиши
stop_thread = threading.Thread(target=stop_on_keypress, args=(stop_flag,))
stop_thread.start()

# Ожидание завершения потока остановки
stop_thread.join()
