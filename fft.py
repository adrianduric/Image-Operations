import time
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import imageio

def konv_timer(f, n):
    starttid = time.time()

    filter = np.ones((n, n)) / (n*n)
    konv = signal.convolve2d(f, filter)

    stopptid = time.time() - starttid

    return stopptid 

def konv_fft_timer(f, n):
    starttid = time.time()

    filter = np.ones((n, n)) / (n*n)
    f_frekvens = np.fft.fft2(f)
    M, N = f.shape
    filter_frekvens = np.fft.fft2(filter, (M, N))
    konv_frekvens = np.real(np.fft.ifft2(f_frekvens*filter_frekvens))
    stopptid = time.time() - starttid

    return stopptid 


if __name__ == "__main__":
    #Oppgave 1.1
    fig = plt.figure()

    #Leser inn innbilde, og lager transformen til frekvensdomenet
    filename = 'cow.png'
    f = imageio.imread(filename, as_gray = True)
    fig.add_subplot(221)
    plt.imshow(f, cmap = 'gray')
    plt.title(filename)
    f_frekvens = np.fft.fft2(f)
    M, N = f.shape

    #Oppretter filter, og lager transformen til frekvensdomenet med nullpadding
    filter = np.ones((15, 15)) / 225
    filter_frekvens = np.fft.fft2(filter, (M, N))

    #Utfører konvolusjon i bildedomenet
    konv = signal.convolve2d(f, filter)
    fig.add_subplot(222)
    plt.title("Konv. i bildedomenet")
    plt.imshow(konv, cmap = 'gray')

    #Konvolusjon i bildedomenet med parameter "same"
    konv_same = signal.convolve2d(f, filter, 'same')
    fig.add_subplot(223)
    plt.title("Konv. i bildedomenet - same")  
    plt.imshow(konv_same, cmap = 'gray')

    #Utfører punktvis mult. i frekvensdomenet, og transformerer tilbake
    konv_frekvens = np.real(np.fft.ifft2(f_frekvens*filter_frekvens))
    fig.add_subplot(224)
    plt.title("Punktvis mult. i frekvensdomenet")  
    plt.imshow(konv_frekvens, cmap = 'gray')

    plt.show()

    #Oppgave 1.3
    filterstorrelser = range(5, 50, 5)
    stopptider_vanlig = []
    stopptider_frekvens = []

    for i in filterstorrelser:
        stopptider_vanlig.append(konv_timer(f, i))
        stopptider_frekvens.append(konv_fft_timer(f, i))
    
    plt.plot(filterstorrelser, stopptider_vanlig, label='Filtrering ved konvolusjon')
    plt.plot(filterstorrelser, stopptider_frekvens, label='Filtrering ved FFT')
    plt.title("Middelverdifiltrering - Tid per filterstørrelse")
    plt.xlabel("Dimensjoner på filter (n)")
    plt.ylabel("Tid (s)")
    plt.legend()
    plt.show()
#end main