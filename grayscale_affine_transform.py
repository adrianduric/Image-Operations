import numpy as np
import matplotlib.pyplot as plt
import imageio

def lagHistogram(f):
    """
    Funksjonen tar inn et 8-bits innbilde, og returnerer to numpy-arrays: et 
    som spesifiserer de 256 mulige intensitetsverdiene, og et som spesifiserer
    antallet piksler med hver intensitetsverdi.

    Args:
        f: 2D numpy-array som spesifiserer et 8-bits bilde.
    Returns:
        intensiteter: 1D numpy-array med de mulige intensitetene.
        antall_per_intenstitet: 1D numpy-array med antall piksler for hver
        intensitet.
    """
    M, N = f.shape

    intensiteter = np.arange(256)
    antall_per_intensitet = np.zeros(256)

    for i in range(M):
        for j in range(N):
            intensitet = f[i, j]
            antall_per_intensitet[intensitet] += 1

    return intensiteter, antall_per_intensitet 
#end lagHistogram

def graatonetransformasjon(f, ny_middelverdi, nytt_standardavvik):
    """
    Funksjonen tar inn et numpy-array som spesifiserer et 8-bits bilde, og 
    returnerer det samme bildet etter en gråtonetransformasjon, med ny 
    middelverdi og standardavvik som spesifisert ved funksjonskallet. 

    Parametre:
        f: 2D numpy-array som spesifiserer et 8-bits bilde. 
        middelverdi: Ønsket middelverdi på bildet som returneres, etter 
        gråtonetransformasjon.
        standardavvik: Ønsket standardavvik på bildet som returneres, 
        etter gråtonetransformasjon.

    Returverdi:
        f_out: bildet som sendes inn (2D numpy-array), etter 
        gråtonetransformasjon.
    """
    M,N = f.shape #M rader, N kolonner

    middelverdi = 0
    varians = 0

    #Finner middelverdi
    for i in range(M):
        for j in range(N):
            middelverdi += f[i,j]
    middelverdi = middelverdi / (M*N)
    #Finner varians
    for i in range(M):
        for j in range(N):
            varians += (f[i,j] - middelverdi)**2
    varians = varians / (M*N)
    standardavvik = np.sqrt(varians)

    #Deklarerer ønsket varians og middelverdi for utbilde
    m_t = ny_middelverdi
    sigma_t = nytt_standardavvik
    varians_t = sigma_t**2
    #Regner ut transformasjonen
    a_t = sigma_t/standardavvik
    b_t = m_t - middelverdi * a_t

    #Setter pikselverdier for utbildet
    f_out = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            f_out[i,j] = a_t * f[i,j] + b_t
            f_out[i,j] = min(f_out[i,j], 255)
            f_out[i,j] = max(f_out[i,j], 0)

    return f_out.astype(int)
#end graatonetransformasjon

def affintransformasjon(f, g, x1, y1, x2, y2, x3, y3, mapping="frem", interpolasjon="nn"):
    """
    Funksjonen tar inn to 8-bits innbilder som en 2D numpy-array, f, og g.
    f er innbildet som affintransformasjonen skal utføres på, mens g brukes
    for å uttrykke dimensjonene man vil projisere det nye bildet på etter 
    transformasjonen (man kan sende samme bilde som f og g, og få ut et bilde
    etter transformasjonen sammen med samme dimensjoner). I tillegg sendes 
    koordinatene til tre punkter i innbildet som skal brukes som
    fra-punkter. Denne funksjonen holder allerede på koordinatene til
    referansepunkter som er manuelt lest av fra geometrimaske.png, men man
    kunne modifisert funksjonen til å også ta inn referansepunkter i g.
    Funksjonen utfører en affin transformasjon som orienterer de tre
    innsendte fra-punktene etter de tre kjente referansepunktene, og 
    returnerer et utbilde etter den gjennomførte transformasjonen.

    Parametre:
        f: 2D numpy-array som spesifiserer et 8-bits bilde.
        g: 2D numpy-array som spesifiserer et 8-bits bilde.
        x1, x2, x3: x-koordinater til hvert av fra-punktene.
        y1, y2, y3: y-koordinater til hvert av fra-punktene.

    Returverdi:
        f_out: bildet som sendes inn (2D numpy-array), etter en affin 
        transformasjon, projisert i dimensjonene til g.
    """
    #Fra-punkter
    x = np.array([x1, x2, x3])
    y = np.array([y1, y2, y3])

    #Referansepunkter
    xref = np.array([258, 258, 440])
    yref = np.array([169, 340, 256])

    #Kolonne med enere
    enere = np.ones(3)

    #Lager matrise som radreduseres for å løse likningssettet
    A = np.column_stack((x, y, enere))
    a_verdier = np.linalg.solve(A, xref)
    b_verdier = np.linalg.solve(A, yref)
    T = np.array([
        a_verdier,
        b_verdier,
        [0, 0, 1]])
    
    #Henter ut a- og b-koeffisienter
    a0 = a_verdier[0]
    a1 = a_verdier[1]
    a2 = a_verdier[2]
    b0 = b_verdier[0]
    b1 = b_verdier[1]
    b2 = b_verdier[2]

    #Inverterer matrisen for å finne de inverterte koeffisientene
    T_inv = np.linalg.inv(T)
    a0_inv = T_inv[0, 0]
    a1_inv = T_inv[0, 1]
    a2_inv = T_inv[0, 2]
    b0_inv = T_inv[1, 0]
    b1_inv = T_inv[1, 1]
    b2_inv = T_inv[1, 2]

    #Utfører transformasjonen på utbildet
    M, N = g.shape
    M_pre, N_pre = f.shape
    f_out = np.zeros((M,N))
    for i in range(M):
        for j in range(N):   
            if mapping == "frem":
                #Forlengs mapping:
                x = int(round(a0 * i + a1 * j + a2))
                y = int(round(b0 * i + b1 * j + b2))
                if x in range(M) and y in range(N) and i in range(M_pre) and j in range(N_pre):
                    f_out[x, y] = f[i, j]
            else:
                #Baklengs mapping
                x = a0_inv * i + a1_inv * j + a2_inv
                y = b0_inv * i + b1_inv * j + b2_inv        
                if interpolasjon == "nn":
                    #Nærmeste nabo-interpolasjon
                    x = int(round(x))
                    y = int(round(y))
                    if x in range(M_pre) and y in range(N_pre):
                        f_out[i, j] = f[x, y]
                else:
                    #Bilineær interpolasjon
                    x0 = int(min(M - 1, np.floor(x))); y0 = int(min(N - 1, np.floor(y)))
                    x1 = int(min(M - 1, np.ceil(x))); y1 = int(min(N - 1, np.ceil(y)))
                    dx = x - x0
                    dy = y - y0
                    p = f[x0, y0] + (f[x1, y0] - f[x0, y1]) * dx
                    q = f[x0, y1] + (f[x1, y1] - f[x0, y1]) * dx
                    
                    f_out[i, j] = p + (q - p) * dy
                    f_out[i, j] = min(255, f_out[i, j])
                    f_out[i, j] = max(0, f_out[i, j])

    return f_out.astype(int)
#end affintransformasjon

if __name__ == "__main__":
    fig = plt.figure()

    filename = 'portrett.png'
    f = imageio.imread(filename, as_gray = True)
    f = f.astype(int)
    fig.add_subplot(221)
    plt.imshow(f, cmap = 'gray')
    plt.title('Innbilde')

    graa = graatonetransformasjon(f, 127, 64)

    """ 
    #Gråtonetransformasjon med histogrammer:

    f_intensiteter, f_antall = lagHistogram(f)
    fig.add_subplot(223)
    plt.bar(f_intensiteter, f_antall)
    plt.title('Histogram - Innbilde')
    plt.xlabel("Intensitet")
    plt.ylabel("Antall piksler")

    fig.add_subplot(222)
    plt.imshow(graa, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Gråtonetransformasjon')

    graa_intensiteter, graa_antall = lagHistogram(graa)
    fig.add_subplot(224)
    plt.bar(graa_intensiteter, graa_antall)
    plt.title('Histogram - Gråtonetransformasjon')
    plt.xlabel("Intensitet")
    plt.ylabel("Antall piksler")
    """

    #Koordinatene for geometrisk transformasjon er avlest manuelt
    maske = imageio.imread("geometrimaske.png", as_gray = True)
    forlengs = affintransformasjon(graa, maske, 87, 85, 67, 119, 108, 129)
    fig.add_subplot(222)
    plt.imshow(forlengs, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Forlengs mapping')

    maske = imageio.imread("geometrimaske.png", as_gray = True)
    invers_nn = affintransformasjon(graa, maske, 87, 85, 67, 119, 108, 129, "bak")
    fig.add_subplot(223)
    plt.imshow(invers_nn, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Baklengs: nærmeste nabo')

    maske = imageio.imread("geometrimaske.png", as_gray = True)
    invers_bilinear = affintransformasjon(graa, maske, 87, 85, 67, 119, 108, 129, "bak", "bilinear")
    fig.add_subplot(224)
    plt.imshow(invers_bilinear, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Baklengs: bilineær interpolasjon')

    plt.show()