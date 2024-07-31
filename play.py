import numpy as np
import sounddevice as sd


def generate_chirp(f0: float, f1: float, duration: float, fs: int) -> np.ndarray:
    """
    Генерация чирп-сигнала.

    :param f0: Начальная частота (Гц).
    :param f1: Конечная частота (Гц).
    :param duration: Продолжительность сигнала (сек).
    :param fs: Частота дискретизации (Гц).
    :return: Чирп-сигнал как массив numpy.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    chirp_signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2))
    return chirp_signal


def play_audio(signal: np.ndarray, fs: int):
    """
    Воспроизведение аудиосигнала с использованием sounddevice.

    :param signal: Аудиосигнал в виде массива numpy.
    :param fs: Частота дискретизации (Гц).
    """
    # Воспроизведение сигнала
    sd.play(signal, samplerate=fs)
    sd.wait()  # Ожидание завершения воспроизведения


def main():
    # Параметры чирп-сигнала
    f0 = 1000  # Начальная частота (Гц)
    f1 = 5000  # Конечная частота (Гц)
    duration = 5.0  # Длительность (сек)
    fs = 44100  # Частота дискретизации (Гц)

    # Генерация и проигрывание чирп-сигнала
    chirp_signal = generate_chirp(f0, f1, duration, fs)
    play_audio(chirp_signal, fs)


if __name__ == "__main__":
    main()
