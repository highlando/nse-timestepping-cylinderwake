import os

# import dolfin
import numpy as np
import scipy.sparse.linalg as spsla

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu
import sadptprj_riclyap_adi.lin_alg_utils as lau
# import matlibplots.conv_plot_utils as cpu

import clean_tdpnse_scheme as tis
import helpers as hlp

N, Re, scheme, tE = 3, 60, 'CR', .2
# Ntslist = [2**x for x in range(6, 11)]
Ntslist = [2**x for x in range(8, 10)]
Ntsref = 2048
tol = 2**(-16)
tolcor = True
method = 1

svdatapathref = 'data/'
svdatapath = 'data/'

if not os.path.exists(svdatapath):
    os.makedirs(svdatapath)
if not os.path.exists(svdatapathref):
    os.makedirs(svdatapathref)

# ###
# ## get the FEM discretization
# ###

femp, stokesmatsc, rhsmomcont \
    = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                        scheme=scheme, mergerhs=True)
fp = rhsmomcont['fp']
fv = rhsmomcont['fv']
stokesmatsc.update(rhsmomcont)

J, MP = stokesmatsc['J'], stokesmatsc['MP']
Nv = J.shape[1]
Mpfac = spsla.splu(MP)

vp_stokes = lau.solve_sadpnt_smw(amat=stokesmatsc['A'], jmat=J, jmatT=-J.T,
                                 rhsv=fv, rhsp=fp)

getconvvec = hlp.getconvvecfun(**femp)
vfile = 'results/vels'
pfile = 'results/pres'
plotit = hlp.getparaplotroutine(femp=femp, vfile=vfile, pfile=pfile)


def getdatastr(t=None):
    return 'data/itsjustatest{0}'.format(t)
vpstrdct = tis.halfexp_euler_nseind2(getconvfv=getconvvec,
                                     get_datastr=getdatastr,
                                     plotroutine=plotit,
                                     vp_init=vp_stokes, **stokesmatsc)
import ipdb; ipdb.set_trace()

# get the ref trajectories
trange = np.linspace(0., tE, Ntsref+1)
M, A = stokesmatsc['M'], stokesmatsc['A']
JT, J = stokesmatsc['JT'], stokesmatsc['J']
invinds = femp['invinds']
fv, fp = rhsd_stbc['fv'], rhsd_stbc['fp']
ppin = None

snsedict = dict(A=A, J=J, JT=JT, M=M, ppin=ppin, fv=fv, fp=fp,
                V=femp['V'], Q=femp['Q'],
                invinds=invinds, diribcs=femp['diribcs'],
                start_ssstokes=True, trange=trange,
                clearprvdata=False, paraviewoutput=True,
                data_prfx='refveldata/',
                vfileprfx='refveldata/v', pfileprfx='refveldata/p',
                return_dictofpstrs=True, return_dictofvelstrs=True)

vdref, pdref = snu.solve_nse(**snsedict)
