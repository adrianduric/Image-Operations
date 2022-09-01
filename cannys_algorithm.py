import numpy as np
import matplotlib.pyplot as plt
import imageio
from numba import jit

@jit(nopython=True) #Benyttes kun for å kjøre programmet raskere

def konvolusjon(f, h):
    """
    Funksjonen tar inn et bilde f og et filter h, og utfører konvolusjon med
    bildet og filteret. Den returnerer det samme bildet etter utført konvolusjon.

    Parametre:
        f: 8-bits 2D numpy-array som spesifiserer et bilde.
        h: 2D numpy-array med odde lengder.

    Returverdi:
        f_out: utbildet, etter at det har blitt utført konvolusjon.
    """
    M, N = f.shape
    h = np.rot90(np.rot90(h))
    m, n = h.shape
    f_out = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            #sum uttrykker en f[i, j] sin nye verdi etter konvolusjon.
            sum = 0
            for s in range(int(-m/2), int(m/2 + 1)):
                for t in range(int(-n/2), int(n/2 + 1)):
                    #Sikrer at x og y ikke indekserer til utenfor bildet, ved å
                    #avrunde dem til nærmeste indeks innenfor (og dermed
                    #avrunde dem til nærmeste pikselverdi).
                    x = min(i - s, M - 1)
                    x = max(x, 0)
                    y = min(j - t, N - 1)
                    y = max(y, 0)

                    a = int(s + m/2)
                    b = int(t + n/2)

                    #Regner verdien i punktet, eller et av punktene rundt, der
                    #vi nå konvolverer (f[i, j]).
                    sum += f[x, y] * h[a, b]
            #Tillegger den nye verdien til pikselen vi utførte konvolusjon på,
            #i utbildet.
            f_out[i, j] = int(round(sum))
    return f_out
#end konvolusjon

def gaussfilter(sigma):
    """
    Funksjonen lager et Gauss-filter av et nxn-naboskap, der størrelsen n og
    standardavviket bestemmes av argumentet sigma.

    Parametre:
        sigma: tallverdi (int/float) som representerer standardavvik og
        bestemmer mengden glatting med filteret, samt indirekte størrelsen
        på naboskapet. Normalverdi: sigma=1.

    Returverdi:
        g: 2D nxn numpy-array som utgjør et Gauss-filter.
    """
    n = int(1 + np.ceil(8*sigma))
    g = np.zeros((n, n))
    sum = 0 #sum brukes til å senere regne ut hva konstanten A blir.
    for i in range(n): #i itererer gjennom rader
        for j in range(n): #j itererer gjennom kolonner
            x_2 = (int(n/2) - i)**2
            y_2 = (int(n/2) - j)**2
            g[i, j] = np.exp(-(x_2 + y_2) / (2*(sigma**2)))
            sum += g[i, j]
    A = 1/sum
    g = A * g
    return g
#end gaussfilter

def gradient(f):
    """
    Funksjonen detekterer gradientmagnitude- og retning i det innsendte
    bildet f, og returnerer to matriser i samme dimensjoner som f med 
    magnitude og retning i hvert punkt.

    Parameter:
        f: 2D numpy-array med intensitetsverdier.

    Returverdier:
        grad_magnitude: 2D numpy-array med gradientmagnitude.
        grad_retning: 2D numpy-array med gradientretning i grader,
        avrundet til nærmeste 45 graders retning.
    """
    M, N = f.shape
    sym_1d_x = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, -1, 0]
    ])
    sym_1d_y = np.array([
        [0, 0, 0],
        [1, 0, -1],
        [0, 0, 0]
    ])

    grad_x = konvolusjon(f, sym_1d_x)
    grad_y = konvolusjon(f, sym_1d_y)
    grad_x_2 = np.square(grad_x)
    grad_y_2 = np.square(grad_y)

    grad_magnitude = np.sqrt(grad_x_2 + grad_y_2)
    grad_retning = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            theta = np.arctan2(grad_y[i, j], grad_x[i, j])
            theta_mod = (round(theta*180/(45*np.pi))*45) % 180
            grad_retning[i, j] = theta_mod
    return grad_magnitude, grad_retning
#end gradient  

def tynning(grad_magnitude, grad_retning):
    """
    Funksjonen tynner de detekterte kantene fra gradientmagnitudeverdiene
    oppgitt i grad_magnitude, ved å sjekke om magnituden i punkt [i, j] er
    større i en av de to pikslene i 8-naboskapet som angis av retningen
    på gradienten, som hentes fra grad_retning.

    Parametre:
        grad_magnitude: 2D numpy-array med gradientmagnitude.
        grad_retning: 2D numpy-array med gradientretning i grader,
        avrundet til nærmeste 45 graders retning.

    Returverdi:
        fortynnet: 2D numpy-array med gradientmagnitude, etter
        fortynning.
    """
    M, N = grad_magnitude.shape
    fortynnet = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            theta = grad_retning[i, j]
            k = grad_magnitude[i, j]
            iminus = min(i - 1, M - 1)
            ipluss = min(i + 1, M - 1)
            jminus = min(j - 1, N - 1)
            jpluss = min(j + 1, N - 1)

            if theta == 0:
                m1 = grad_magnitude[iminus, j]
                m2 = grad_magnitude[ipluss, j]
                if not (m1 > k or m2 > k):
                    fortynnet[i, j] = grad_magnitude[i, j]
            elif theta == 45:
                m1 = grad_magnitude[iminus, jminus]
                m2 = grad_magnitude[ipluss, jpluss]
                if not (m1 > k or m2 > k):
                    fortynnet[i, j] = grad_magnitude[i, j]
            elif theta == 90:
                m1 = grad_magnitude[i, jminus]
                m2 = grad_magnitude[i, jpluss]
                if not (m1 > k or m2 > k):
                    fortynnet[i, j] = grad_magnitude[i, j]
            elif theta == 135:
                m1 = grad_magnitude[ipluss, jminus]
                m2 = grad_magnitude[iminus, jpluss]
                if not (m1 > k or m2 > k):
                    fortynnet[i, j] = grad_magnitude[i, j]
    return fortynnet
#end tynning

def terskling(f, t_h, t_l):
    """
    Funksjonen filtrerer ut verdier i kantbildet f som spesifisert av
    terskelverdiene t_h og t_l. Alle verdier høyere enn t_h blir tatt med
    i utbildet f_tersklet. Alle verdier mellom t_h og t_l blir tatt med,
    dersom de grenser til en piksel med verdi større enn t_h, eller til
    en piksel mellom t_h og t_l som allerede grenser til en annen med
    verdi større enn t_h, osv.

    Parametre:
        f: 2D numpy-array med gradientmagnitudeverdier.
        t_h: Høy terskel, som spesifiserer hvilke gradientmagnituder
        som alltid blir inkludert i f_tersklet.
        t_l: Lav terskel, som spesifiserer hvilke gradientmagnituder
        som blir inkludert i f_tersklet dersom de grenser til piksler
        som allerede blir inkludert.

    Returverdi:
        f_tersklet: 2D numpy-array med gradientmagnitudeverdier, etter
        terskling.
    """
    M, N = f.shape
    f_tersklet = np.zeros((M, N))
    endring = 100
    antall_loops = 0
    while endring > 0: 
        """
        Merk: For å øke kjøretiden og uten å endre utbildet nevneverdig,
        kan man endre hvor mange endringer som må skje per loop til noe
        høyere, f.eks. 'while endring > 10:', for å drastisk senke
        antall loops (dette avhenger av innsendte parametre i Cannys-
        funksjonen).
        """
        endring = 0
        for i in range(M):
            for j in range(N):
                if f[i, j] >= t_h:
                    f_tersklet[i, j] = 255
                elif f[i, j] >= t_l and f_tersklet[i, j] != 255:
                    iminus = min(i - 1, M - 1)
                    iminus = max(0, iminus)
                    ipluss = min(i + 2, M - 1)
                    ipluss = max(0, ipluss)
                    jminus = min(j - 1, N - 1)
                    jminus = max(0, jminus)
                    jpluss = min(j + 2, N - 1)
                    jpluss = max(0, jpluss)
                    for x in range(iminus, ipluss):
                        for y in range(jminus, jpluss):
                            if f_tersklet[x, y] == 255:
                                f_tersklet[i, j] = 255
                                endring += 1
    return f_tersklet
#end terskling

def canny(f, sigma, t_h, t_l):
    """
    Funksjonen utfører den fulle implementasjonen av Cannys algoritme på
    innbildet f, med brukerspesifisert standardavvik og terskelverdier.

    Parametre:
        f: 2D numpy-array med innbildet.
        t_h: Høy terskel, som spesifiserer hvilke gradientmagnituder
        som alltid blir inkludert i f_tersklet.
        t_l: Lav terskel, som spesifiserer hvilke gradientmagnituder
        som blir inkludert i f_tersklet dersom de grenser til piksler
        som allerede blir inkludert.

    Returverdi:
        tersklet: 2D numpy-array med detekterte kanter fra innbildet f.
    """
    f_utjevnet = konvolusjon(f, gaussfilter(sigma))
    grad_m, grad_r = gradient(f_utjevnet)
    kanter = tynning(grad_m, grad_r)
    tersklet = terskling(kanter, t_h, t_l)

    return tersklet.astype(int)
#end cannys


if __name__ == "__main__":
    fig = plt.figure()

    filename = 'cellekjerner.png'
    f = imageio.imread(filename, as_gray = True)
    f = f.astype(int)

    f_out = canny(f, 3, 20, 10)
    fig.add_subplot(111)
    plt.imshow(f_out, cmap = 'gray')
    plt.title('Detekterte kanter')

    plt.show()
#end main