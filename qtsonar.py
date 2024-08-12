import sys
import select
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from sonar_io import RealTimeSonar
import numpy as np

def visualize(sonar):
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Sonar Distance Visualization")
    win.resize(800, 300)
    win.setWindowTitle("Sonar Distance")
    plot = win.addPlot(title="Distance")
    bar = pg.BarGraphItem(x=[0], height=[0], width=0.8)
    plot.addItem(bar)

    def update():
        if not sonar.Qdata.empty():
            new_line = sonar.Qdata.get()
            if new_line is not None:
                distance = np.argmax(new_line) / sonar.Nplot * sonar.maxdist
                bar.setOpts(x=[0], height=[distance])
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key:
                sonar.stop_flag.set()
                timer.stop()
                app.quit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    app.exec_()

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

if __name__ == "__main__":
    sonar = RealTimeSonar(f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature)
    threads = sonar.start()

    visualize(sonar)

    sonar.stop_flag.set()
    sonar.join_threads(threads)
