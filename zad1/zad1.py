import sys 
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyzeArguments(): #handling command line arguments
    if len(sys.argv)!=4:
        print('Nieprawidlowa liczba argumentow!')
        sys.exit()
    return sys.argv[1:]

def loadParameters(fileName): #loading parameters from file
    parameters = []
    with open(fileName) as f:
        for line in f:
             line = line.split('#', 1)[0]
             parameters.append(float(line))
    return parameters
    
def initializeB(a): #creating unit cell vectors
    b0 = np.array([a, 0, 0])
    b1 = np.array([a/2, a*np.sqrt(3)/2, 0])
    b2 = np.array([a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)])
    return b0, b1, b2
    
def initizalizeR(n, N, b0, b1, b2):
    r = np.zeros([N, 3])
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                i = i0 + i1*n+i2*n*n
                r[i, :] = (i0-(n-1)/2)*b0 + (i1-(n-1)/2)*b1 + (i2-(n-1)/2)*b2
                
    return r

def xyzFileCreator(xyzFileName, r, N):
    with open(xyzFileName, 'w') as f:
        f.write(f'{N}\n\n')
        for i in range(N):
            f.write(f'Ar\t{r[i, 0]}\t{r[i, 1]}\t{r[i, 2]}\n')

def randomEnergy(T0):
    k = 8.31e-3
    x = random.random()
    if x == 0.0:
        x = 1.
    return -k*T0*np.log(x)/2
   
def randomSign():
    return random.randint(0, 1)*2-1
   
def vectorNormSquared(x):
    return x.dot(x)
    
def vectorNorm(x):
    return np.sqrt(vectorNormSquared(x))
    
def initializeP(N, m, T0):
    p = np.zeros([N, 3])
    for i in range(N):
        for j in range(3):
            p[i, j] = randomSign()*np.sqrt(2*m*randomEnergy(T0))   
    p = p - sum(p)/N
    return p

def histogramsP(p):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(p[:, 0], 15, color='green')
    ax2.hist(p[:, 1], 15, color='green')
    ax3.hist(p[:, 2], 15, color='green')
    
    fig.suptitle('Histogramy pędów')
    ax1.set_xlabel('p_x')
    ax2.set_xlabel('p_y')
    ax3.set_xlabel('p_z')
    
    plt.show()
 
def calculateVP(e, R, ri, rj):
    rij = vectorNorm(ri-rj)
    return e*((R/rij)**12-2*(R/rij)**6)
    
def calculateVS(L, f, ri):
    ri_norm = vectorNorm(ri)
    if ri_norm >= L:
        return f/2*(ri_norm-L)**2
    return 0
    
def calculateFP(e, R, ri, rj):
    rij = vectorNorm(ri-rj)
    return 12*e*((R/rij)**12-(R/rij)**6)*(ri-rj)/vectorNormSquared(ri-rj)
    
def calculateFS(L, f, ri):
    ri_norm = vectorNorm(ri)
    if ri_norm >= L:
        return f*(L-ri_norm)*ri/ri_norm
    return np.zeros(3)

def calculateFPV(N, e, R, L, f, r):
    F = np.zeros([N, 3])
    P = 0
    V = 0
    for i in range(N):
        ri = r[i, :]
        V = V + calculateVS(L, f, ri)
        FSi = calculateFS(L, f, ri)
        F[i, :] = F[i, :] + FSi
        P = P + vectorNorm(FSi)
        for j in range(i):
            rj = r[j, :]
            V = V + calculateVP(e, R, ri, rj)
            Fij = calculateFP(e, R, ri, rj)
            F[i, :] = F[i, :] + Fij
            F[j, :] = F[j, :] - Fij
    P = P/(4*np.pi*L**2)
    return F, P, V

def Ekin_i(pi, m):
    return vectorNormSquared(pi)/(2*m)
    
def Ekin(p, m):
    E = 0
    for pi in p:
        E = E + vectorNormSquared(pi)
    return E/(2*m)

def integrate(outputFileName, xyzFileName, p, r, F, P, V, tau, m, L, N, e, R, f, So, Sd, Sout, Sxyz):
    outputFile = open(outputFileName, 'w')
    xyzFile = open(xyzFileName, 'w')
    
    k = 8.31e-3
    
    Tav = 0
    Pav = 0
    Hav = 0
    
    T = 0
    H = 0
    for s in tqdm(range(So+Sd)):
        p = p + F*tau/2
        r = r + p*tau/m
        F, P, V = calculateFPV(N, e, R, L, f, r)
        p = p + F*tau/2
        E_kin = Ekin(p, m)
        T = 2*E_kin/(3*N*k)
        H = E_kin + V
        if not s%Sout:
            t = s*tau
            #zapis t, H, V, T, P
            outputFile.write(f'{t}\t{H}\t{V}\t{T}\t{P}\n')
        if not s%Sxyz:   
            #zapis x, y, z E_kin dla każdego atomu do avs.dat
            xyzFile.write(f'{N}\n\n')
            for i in range(N):
                xyzFile.write(f'Ar\t{r[i, 0]}\t{r[i, 1]}\t{r[i, 2]}\t{Ekin_i(p[i, :], m)}\n')
        if s>=So:
            #akumulacja average
            Tav = Tav + T
            Pav = Pav + P
            Hav = Hav + H      
    #normalizacja
    Tav = Tav/Sd
    Pav = Pav/Sd
    Hav = Hav/Sd
    
    outputFile.close()
    xyzFile.close()
    
def main():
    parametersFileName, outputFileName, xyzFileName = analyzeArguments()
    
    #loading parameters
    n, m, e, R, f, L, a, T0, tau, So, Sd, Sout, Sxyz = loadParameters(parametersFileName)
    n = int(n)
    N = n**3
    So = int(So)
    Sd = int(Sd)
    Sout = int(Sout)
    Sxyz = int(Sxyz)
    
    b0, b1, b2 = initializeB(a)
    r = initizalizeR(n, N, b0, b1, b2)
    #xyzFileCreator(xyzFileName, r, N)
    p = initializeP(N, m, T0)
    #histogramsP(p)
    
    F, P, V = calculateFPV(N, e, R, L, f, r)
    print("Sily:")
    print(F)
    print("Cisnienie:", P)
    print("Energia potencjalna:", V)
    H0 = Ekin(p, m) + V
    print("Energia calkowita:", H0)
    integrate(outputFileName, xyzFileName, p, r, F, P, V, tau, m, L, N, e, R, f, So, Sd, Sout, Sxyz)
main()