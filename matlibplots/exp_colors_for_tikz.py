import matplotlib
import numpy as np

N = 6
print "Don't know why this only works in IPython"
cm = matplotlib.cm.cool(np.linspace(0, 1, N))
# cm = matplotlib.cm.cubehelix(np.linspace(0, 1, N))

# cm = np.round(255*cm).astype(int)

fc = open('coolfrommpl.gpl', 'w')
fc.write('GIMP Palette\nName: Cool\nColumns: 0\n#\n')

for k in range(N):
    cmk = cm[k]
    fc.write('\definecolor{color' + '{0}'.format(k) + '}{rgb}{' +
             '{0}, {1}, {2}}}\n'.format(cmk[0], cmk[1], cmk[2]))

fc.close()
