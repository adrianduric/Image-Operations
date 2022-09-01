import numpy as np
import matplotlib.pyplot as plt
import imageio
from numba import jit
import os

@jit(nopython=True)

def c(a):
    """
    Hjelpefunksjon for 2D DCT/IDCT. Bestemmer c-verdien avhengig av a.

    Parametre:
        a: koordinat som brukes i 2D DCT/IDCT (u eller v).

    Returverdi:
        c: Enten 1/sqrt(2) eller 1, avhengig av a.
    """
    return 1/np.sqrt(2) if a == 0 else 1

@jit(nopython=True)

def dct_2d(f):
    """
    Funksjonen gjennomfører en 2D diskret cosinustransformasjon på innbildet f.

    Parametre:
        f: 2D numpy-array som representerer et bilde.

    Returverdi:
        F: 2D numpy-array som representerer en 2D DCT av innbildet.
    """
    f -= 128
    M, N = f.shape
    F = np.zeros((M, N))
    for i in range(0, M, 8): #Inkrementerer med 8 for å lese ut blokker
        for j in range(0, N, 8):
            for u in range(8):
                for v in range(8):
                    F_sum = 0
                    for x in range(8):
                        for y in range(8):
                            F_sum += f[i+x, j+y]*np.cos(((2*x+1)*u*np.pi)/16)*np.cos(((2*y+1)*v*np.pi)/16)
                    F[i+u, j+v] = 0.25*c(u)*c(v)*F_sum
    return F

@jit(nopython=True)

def idct_2d(F):
    """
    Funksjonen gjennomfører en invers 2D diskret cosinustransformasjon på innbildet f.

    Parametre:
        F: 2D numpy-array som representerer en 2D DCT av et bilde.

    Returverdi:
        f: 2D numpy-array som representerer det opprinnelige bildet.
    """
    M, N = F.shape
    f = np.zeros((M, N))
    for i in range(0, M, 8): #Inkrementerer med 8 for å lese ut blokker
        for j in range(0, N, 8):
            for x in range(8):
                for y in range(8):
                    f_sum = 0
                    for u in range(8):
                        for v in range(8):
                            f_sum += c(u)*c(v)*F[i+u, j+v]*np.cos(((2*x+1)*u*np.pi)/16)*np.cos(((2*y+1)*v*np.pi)/16)
                    f[i+x, j+y] = np.round(0.25*f_sum) + 128
    return f

def kvantisering_div(F, q, Q):
    """
    Funksjonen punktvis dividerer verdiene i F, på punktverdiene i Q
    multiplisert med q.

    Parametre:
        F: 2D numpy-array som representerer en 2D DCT av et bilde.
        q: Tallverdi.
        Q: 2D numpy-array som representerer en kvantifiseringsmatrise.

    Returverdi:
        F: Inputverdien F etter den punktvise divisjonen.
    """
    M, N = F.shape
    for i in range(0, M, 8): #Inkrementerer med 8 for å lese ut blokker
        for j in range(0, N, 8):
            F[i:(i+8), j:(j+8)] = np.round(F[i:(i+8), j:(j+8)] / (q*Q))
    return F

def kvantisering_mult(F, q, Q):
    """
    Funksjonen punktvis multipliserer verdiene i F, med punktverdiene i Q
    multiplisert med q.

    Parametre:
        F: 2D numpy-array som representerer en 2D DCT av et bilde.
        q: Tallverdi.
        Q: 2D numpy-array som representerer en kvantifiseringsmatrise.

    Returverdi:
        F: Inputverdien F etter den punktvise multiplikasjonen.
    """
    M, N = F.shape
    for i in range(0, M, 8): #Inkrementerer med 8 for å lese ut blokker
        for j in range(0, N, 8):
            F[i:(i+8), j:(j+8)] = np.round(F[i:(i+8), j:(j+8)] * (q*Q))
    return F

def normHistogram(f):
    """
    Funksjonen mapper hver verdi i innbildet f til en teller i et dict,
    og teller hvor mange ganger hver verdi forekommer, og deler denne
    verdien på antallet verdier totalt.

    Parametre:
        f: 2D numpy-array som spesifiserer et bilde.

    Returverdi:
        intensiteter: dict med de mulige intensitetene som nøkler, og
        antallet forekomster av hver intensitet som verdier, normalisert.
    """
    M, N = f.shape

    intensiteter = {}

    for i in range(M):
        for j in range(N):
            if (f[i, j] in intensiteter.keys()):
                intensiteter[f[i, j]] += 1
            else:
                intensiteter[f[i, j]] = 1

    for key in intensiteter:
        intensiteter[key] = intensiteter[key] / (M*N)

    return intensiteter

def entropi(F):
    """
    Funksjonen beregner entropien til en 2D DCT av et bilde basert på
    DCT-ens normaliserte histogram.

    Parametre:
        F: 2D numpy-array som spesifiserer en 2D DCT av et bilde, etter
        punktvis divisjon med kvantifiseringsmatrise.

    Returverdi:
        H: tallverdi på entropien til f.
    """
    H = 0

    intensiteter = normHistogram(F)

    for key in intensiteter:
        H -= intensiteter[key] * np.log2(intensiteter[key])

    return H

def kompresjonsrate(f, F):
    """
    Funksjonen sammenligner entropien til innbildet f mot entropien
    til bildet F, og beregner kompresjonsraten deretter.

    Parametre:
        f: 2D numpy-array som spesifiserer et bilde.
        F: 2D numpy-array som spesifiserer en 2D DCT av et bilde, etter
        punktvis divisjon med kvantifiseringsmatrise.

    Returverdi:
        CR: kompresjonsraten ved komprimering fra f til F.
    """
    b = entropi(f)
    c = entropi(F)

    CR = b/c

    return CR

def kompresjon(filnavn, q):
    """
    Funksjonen utfører kompresjon av bildet spesifisert ved filnavn, med
    kompresjonsrate som indirekte bestemmes av q. Den skriver ut det
    komprimerte bildet på skjermen, samt skriver ut forventet lagringsplass
    for det komprimertet bildet.

    Parametre:
        filnavn: string som spesifiserer navnet på bildefilen som brukes.
        q: int som indirekte bestemmer kompresjonsraten.
    """
    #Steg 1
    f_in = imageio.imread(filnavn, as_gray = True) #Lagres til steg 6
    f = f_in

    fig = plt.figure()

    fig.add_subplot(231)
    plt.imshow(f, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Innbilde')

    #Steg 2 og 3
    F = dct_2d(f)

    fig.add_subplot(232)
    plt.imshow(F, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('2D DCT')

    #Steg 4
    f = idct_2d(F)

    fig.add_subplot(233)
    plt.imshow(f, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('2D IDCT')

    #Steg 5
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    F = kvantisering_div(F, q, Q)

    fig.add_subplot(234)
    plt.imshow(F, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('2D DCT pkt.vis dividert')

    #Steg 6
    CR = np.round(kompresjonsrate(f_in, F), 2)
    filstorrelse = int(os.path.getsize(filnavn) / CR)
    print("Forventet filstørrelse for", filnavn, "med kompresjonsrate ",
        CR, ": ", filstorrelse, "B")

    #Steg 7
    F = kvantisering_mult(F, q, Q)

    fig.add_subplot(235)
    plt.imshow(F, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('2D DCT pkt.vis multiplisert')

    f = idct_2d(F)

    nytt_filnavn = filnavn[:-4] + "_komprimert_" + str(q) + ".png"
    plt.imsave(nytt_filnavn, f, format = 'png', cmap = 'gray')

    fig.add_subplot(236)
    plt.imshow(f, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Innbilde rekonstruert: q = ' + str(q))

    plt.show()

if __name__ == "__main__":
    kompresjon("uio.png", 0.1)
    kompresjon("uio.png", 0.5)
    kompresjon("uio.png", 2)
    kompresjon("uio.png", 8)
    kompresjon("uio.png", 32)
