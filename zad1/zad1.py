import time
start = time.time() 
import sys 
import numpy as np
#import matplotlib.pyplot as plt #uncomment to draw histograms
from tqdm import tqdm
from numba import jit

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

@jit    
def initizalizeR(n, N, b0, b1, b2): #creating and initializing position vectors
    r = np.zeros((N, 3))
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                i = i0 + i1*n+i2*n*n
                r[i] = (i0-(n-1)/2)*b0 + (i1-(n-1)/2)*b1 + (i2-(n-1)/2)*b2
    return r

@jit
def randomEnergy(T0, N): #generating random energy from Maxwell distribution
    x = np.random.random((N, 3))
    for xi in x:
        for xj in xi:
            if xj == 0.0:
                xj = 1.
    return -8.31e-3*T0*np.log(x)/2
   
def randomSign(N): #generating randomly + or -
    return np.random.randint(2, size=(N, 3))*2-1

    
def initializeP(N, m, T0): #creating and initializing momentum vectors
    p = np.zeros((N, 3))
    p = np.multiply(randomSign(N),  np.sqrt(2*m*randomEnergy(T0, N)))   
    p = p - sum(p)/N
    return p

def histogramsP(p): #drawing initial momentum histograms
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(p[:, 0], 15, color='green')
    ax2.hist(p[:, 1], 15, color='green')
    ax3.hist(p[:, 2], 15, color='green')
    
    fig.suptitle('Histogramy pędów')
    ax1.set_xlabel('p_x')
    ax2.set_xlabel('p_y')
    ax3.set_xlabel('p_z')
    
    plt.show()

@jit 
def calculateVP(e, R, ri, rj): #calculating Vp
    r_diff = ri-rj
    rij = np.sqrt(r_diff.dot(r_diff))
    z = R/rij
    return e*((z)**12-2*(z)**6)
    
@jit   
def calculateVS(L, f, ri): #calculating Vs
    ri_norm = np.sqrt(ri.dot(ri))
    if ri_norm >= L:
        return f/2*(ri_norm-L)**2
    return 0
    
@jit    
def calculateFP(e, R, ri, rj): #calculating Fp
    r_diff = ri-rj
    rij2 = r_diff.dot(r_diff)
    rij = np.sqrt(rij2)
    z = R/rij
    return 12*e*((z)**12-(z)**6)*(r_diff)/rij2

@jit    
def calculateFS(L, f, ri): #calculating Fs
    ri_norm = np.sqrt(ri.dot(ri))
    if ri_norm >= L:
        return f*(L-ri_norm)*ri/ri_norm
    return np.zeros(3)

@jit
def calculateFPV(N, e, R, L, f, r): #calculating F, P, V
    F = np.zeros((N, 3))
    P = 0
    V = 0
    for i in range(N):
        ri = r[i]
        V = V + calculateVS(L, f, ri)
        FSi = calculateFS(L, f, ri)
        F[i] = np.add(F[i], FSi)
        P = P + np.sqrt(FSi.dot(FSi))
        for j in range(i):
            rj = r[j]
            V = V + calculateVP(e, R, ri, rj)
            Fij = calculateFP(e, R, ri, rj)
            F[i] = np.add(F[i], Fij)
            F[j] = np.subtract(F[j], Fij)
    P = P/(4*np.pi*L**2)
    return F, P, V

def Ekin_i(pi, m): #calculating kinetic energy
    return pi.dot(pi)/(2*m)
    
def Ekin(p, m): #calculating total kinetic energy
    E = 0
    for pi in p:
        E = E + pi.dot(pi)
    return E/(2*m)

def integrate(outputFileName, xyzFileName, p, r, F, P, V, tau, m, L, N, e, R, f, So, Sd, Sout, Sxyz): #simulation
    outputFile = open(outputFileName, 'w')
    xyzFile = open(xyzFileName, 'w')
    
    k = 8.31e-3
    
    Tav = 0
    Pav = 0
    Hav = 0
    
    T = 0
    H = 0
    for s in tqdm(range(So+Sd)):
        p = np.add(p, np.multiply(F, tau/2))
        r = np.add(r, np.multiply(p, tau/m))
        F, P, V = calculateFPV(N, e, R, L, f, r)
        p = np.add(p, np.multiply(F, tau/2))
        E_kin = Ekin(p, m)
        T = 2*E_kin/(3*N*k)
        H = E_kin + V
        if not s%Sout:
            t = s*tau
            #saving t, H, V, T, P to file
            outputFile.write(f'{t}\t{H}\t{V}\t{T}\t{P}\n')
        if not s%Sxyz:   
            #daving x, y, z E_kin to file
            xyzFile.write(f'{N}\n\n')
            for i in range(N):
                xyzFile.write(f'Ar\t{r[i, 0]}\t{r[i, 1]}\t{r[i, 2]}\t{Ekin_i(p[i], m)}\n')
        if s>=So:
            #accumulation average
            Tav = Tav + T
            Pav = Pav + P
            Hav = Hav + H      
    #normalizacja
    Tav = Tav/Sd
    Pav = Pav/Sd
    Hav = Hav/Sd
    
    outputFile.close()
    xyzFile.close()
    
def main(): #main function
    parametersFileName, outputFileName, xyzFileName = analyzeArguments()
    
    #loading parameters
    n, m, e, R, f, L, a, T0, tau, So, Sd, Sout, Sxyz = loadParameters(parametersFileName)
    n = int(n)
    N = n**3
    So = int(So)
    Sd = int(Sd)
    Sout = int(Sout)
    Sxyz = int(Sxyz)
    
    #initializing
    b0, b1, b2 = initializeB(a)
    r = initizalizeR(n, N, b0, b1, b2)
    p = initializeP(N, m, T0)
    #histogramsP(p) #uncomment to draw momentum histograms
    
    F, P, V = calculateFPV(N, e, R, L, f, r)
    #print("Sily:")
    #print(F)
    #print("Cisnienie:", P)
    #print("Energia potencjalna:", V)
    #H0 = Ekin(p, m) + V
    #print("Energia calkowita:", H0)
    integrate(outputFileName, xyzFileName, p, r, F, P, V, tau, m, L, N, e, R, f, So, Sd, Sout, Sxyz)
main()
end = time.time()
print(end-start)