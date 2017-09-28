import os
import glob

# import dolfin
import numpy as np
# import scipy.sparse.linalg as spsla
#
# import dolfin_navier_scipy.problem_setups as dnsps
# import dolfin_navier_scipy.stokes_navier_utils as snu
# import sadptprj_riclyap_adi.lin_alg_utils as lau
# import matlibplots.conv_plot_utils as cpu

import tdp_nse_schemes as tns
import helpers as hlp

N, Re, scheme = 2, 60, 'TH'
# Ntslist = [2**x for x in range(6, 11)]
t0, tE, Nts = 0., 1., 1024  # 2048
linatol = 1e-4
linatol = 0  # 1e-4
trange = np.linspace(t0, tE, Nts)
curmethnm = 'projectn2'
curmethnm = 'imexeuler'
curmethnm = 'SIMPLE'
cleardata = True
plotplease = True

methdict = {'imexeuler': tns.halfexp_euler_nseind2,
            'projectn2': tns.projection2,
            'SIMPLE': tns.SIMPLE}
curmethod = methdict[curmethnm]

svdatapathref = 'data/'
svdatapath = 'data/'

if not os.path.exists(svdatapath):
    os.makedirs(svdatapath)
if not os.path.exists(svdatapathref):
    os.makedirs(svdatapathref)

# ###
# ## get the FEM discretization
# ###

coeffs = hlp.getthecoeffs(N=N, Re=Re, scheme=scheme)

vfile = 'plots/justatest_' + curmethnm + 'vels'
pfile = 'plots/justatest_' + curmethnm + 'pres'
plotit = hlp.getparaplotroutine(femp=coeffs['femp'], plotplease=plotplease,
                                vfile=vfile, pfile=pfile)

parastr = 'Re{0}N{1}scheme{5}Nts{2}t0{3}tE{4}linatol{6:.2e}'.\
    format(Re, N, Nts, t0, tE, scheme, linatol)


def getdatastr(t=None):
    return 'data/testit_' + curmethnm + parastr +\
        'linatol{0:.2e}'.format(linatol) + 't{0:.5f}'.format(t)

if cleardata:
    cdatstr = 'data/testit_' + curmethnm + parastr + '*'
    for fname in glob.glob(cdatstr):
        os.remove(fname)

curslvdct = coeffs
curslvdct.update(trange=trange, get_datastr=getdatastr,
                 plotroutine=plotit, numoutputpts=100, linatol=linatol,
                 getconvfv=coeffs['getconvvec'])

vdc, pdc = curmethod(**curslvdct)

# vpstrdct = tns.halfexp_euler_nseind2(getconvfv=getconvvec,
#                                      get_datastr=getdatastr,
#                                      plotroutine=plotit,
#                                      vp_init=vp_stokes, **stokesmatsc)

# import ipdb; ipdb.set_trace()
#
# # get the ref trajectories
# trange = np.linspace(0., tE, Ntsref+1)
# M, A = stokesmatsc['M'], stokesmatsc['A']
# JT, J = stokesmatsc['JT'], stokesmatsc['J']
# invinds = femp['invinds']
# fv, fp = rhsd_stbc['fv'], rhsd_stbc['fp']
# ppin = None
#
# snsedict = dict(A=A, J=J, JT=JT, M=M, ppin=ppin, fv=fv, fp=fp,
#                 V=femp['V'], Q=femp['Q'],
#                 invinds=invinds, diribcs=femp['diribcs'],
#                 start_ssstokes=True, trange=trange,
#                 clearprvdata=False, paraviewoutput=True,
#                 data_prfx='refveldata/',
#                 vfileprfx='refveldata/v', pfileprfx='refveldata/p',
#                 return_dictofpstrs=True, return_dictofvelstrs=True)
#
# vdref, pdref = snu.solve_nse(**snsedict)
