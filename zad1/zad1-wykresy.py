import matplotlib.pyplot as plt
import sys

def analyzeArguments(): #handling command line arguments
    if len(sys.argv)!=2:
        print('Nieprawidlowa liczba argumentow!')
        sys.exit()
    return sys.argv[1]
    
outputFileName = analyzeArguments()
t = []
H = []
V = []
T = []
P = []
with open(outputFileName) as f:
    for line in f:
        line = line.split()
        t.append(float(line[0]))
        H.append(float(line[1]))
        V.append(float(line[2]))
        T.append(float(line[3]))
        P.append(float(line[4]))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(t, H)
ax1.plot(t, V)
ax2.plot(t, T)
ax3.plot(t, P)

ax1.set_xlabel('t (ps)')
ax2.set_xlabel('t (ps)')
ax3.set_xlabel('t (ps)')
 
ax1.set_ylabel('H, V (kJ/mol)')
ax2.set_ylabel('T (K)')
ax3.set_ylabel('P (16,6 atm.)')
 
ax1.legend(('H', 'V'))

ax1.grid()
ax2.grid()
ax3.grid()

plt.subplots_adjust(hspace = 0.5)
plt.show()
