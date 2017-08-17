import os

import dolfin
import numpy as np
# from time_int_schemes import expand_vp_dolfunc, get_dtstr

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.dolfin_to_sparrays as dts

# import matlibplots.conv_plot_utils as cpu

import tdp_nse_schemes as tis
import helpers as hlp

dolfin.set_log_level(60)

samplerate = 1

Nref = 3
N, Re, scheme, t0, tE = 3, 60, 'CR', 0., .01
# Ntslist = [2**x for x in range(6, 11)]
Ntslist = [12]  # 2**x for x in range(5, 6)]
Ntsref = 12
tol = 2**(-16)
tolcor = True
method = 1

datapathref = 'refdata/'
datapath = 'data/'
plotspath = 'plots/'

if not os.path.exists(datapath):
    os.makedirs(datapath)
if not os.path.exists(datapathref):
    os.makedirs(datapathref)
if not os.path.exists(plotspath):
    os.makedirs(plotspath)

# ###
# ## get the FEM discretization
# ###


refcoeffs = hlp.getthecoeffs(N=Nref, Re=Re)

# get the ref trajectories
trange = np.linspace(t0, tE, Ntsref+1)

snsedict = dict(A=refcoeffs['A'], J=refcoeffs['J'], JT=refcoeffs['JT'],
                M=refcoeffs['M'], ppin=refcoeffs['ppin'], fv=refcoeffs['fv'],
                fp=refcoeffs['fp'], V=refcoeffs['V'], Q=refcoeffs['Q'],
                invinds=refcoeffs['invinds'], diribcs=refcoeffs['diribcs'],
                iniv=refcoeffs['iniv'], trange=trange,
                nu=refcoeffs['femp']['nu'],
                clearprvdata=False, paraviewoutput=True,
                nsects=1, addfullsweep=True,
                vel_pcrd_stps=1,
                data_prfx=datapathref,
                vfileprfx=plotspath+'refv_',
                pfileprfx=plotspath+'refp_',
                return_dictofpstrs=True, return_dictofvelstrs=True)

vdref, pdref = snu.solve_nse(**snsedict)

Nts = Ntslist[0]
coeffs = hlp.getthecoeffs(N=N, Re=Re)
parastr = 'Re{0}N{1}Nts{2}t0{3}tE{4}'.format(Re, N, Nts, t0, tE)


def getdatastr(t=None):
    return datapath + 'imexeuler' + parastr + 't{0:.5f}'.format(t)

vfile = plotspath + 'imexeulervels'
pfile = plotspath + 'imexeulerpres'
plotit = hlp.getparaplotroutine(femp=coeffs['femp'], vfile=vfile, pfile=pfile)
tsdict = dict(t0=t0, tE=tE, Nts=Nts)
curslvdct = coeffs
curslvdct.update(tsdict)
vstrdct, pstrdct = tis.halfexp_euler_nseind2(get_datastr=getdatastr,
                                             plotroutine=plotit,
                                             getconvfv=coeffs['getconvvec'],
                                             **curslvdct)

NV, NP = coeffs['J'].T.shape
refiniv = dou.load_npa(vdref[trange[0]])
imxiniv = dou.load_npa(vstrdct[trange[0]])

refinip = dou.load_npa(pdref[trange[0]])
imxinip = dou.load_npa(pstrdct[trange[0]])
print('ref vs. inival: {0}'.format(np.linalg.norm(refiniv-refcoeffs['iniv'])))
print('imx vs. inival: {0}'.format(np.linalg.norm(imxiniv - coeffs['iniv'])))
print('diff in inivels: {0}'.format(np.linalg.norm(refiniv - imxiniv)))
print('diff in inipres: {0}'.format(np.linalg.norm(refinip - imxinip)))

# import ipdb; ipdb.set_trace()

# errvl = []
# errpl = []


def compvperror(reffemp=None, vref=None, pref=None,
                curfemp=None, vcur=None, pcur=None):
    vreff, preff = dts.expand_vp_dolfunc(vc=vref, pc=pref, **reffemp)
    vcurf, pcurf = dts.expand_vp_dolfunc(vc=vcur, pc=pcur, **curfemp)
    return dolfin.errornorm(vreff, vcurf), dolfin.errornorm(preff, pcurf)

elv = []
elp = []

curttrange = trange
for t in curttrange:
    print(t)

# rescl = []
# for Nts in Ntslist:
#     dtstrdct = dict(prefix=svdatapath, method=method,
#                     Nts=Nts, tol=tol, te=tE, tolcor=tolcor)
#     elc = []
#     def app_pverr(tcur):
#         cdatstr = get_dtstr(t=tcur, **dtstrdct)
#         vp = np.load(cdatstr + '.npy')
#         v, p = expand_vp_dolfunc(vp=vp)
#         # vpref = np.load(cdatstrref + '.npy')
#         # vref, pref = expand_vp_dolfunc(PrP, vp=vpref)
#         vref = np.load(vdref[tcur] + '.npy')
#         pref = np.load(pdref[tcur] + '.npy')
#         # vpref = np.vstack([vref, pref])
#         vreff, preff = expand_vp_dolfunc(vc=vref, pc=pref)
#         # vdiff, pdiff = expand_vp_dolfunc(PrP, vc=vp[:Nv]-vref,
#         #                                  pc=vp[Nv:]-pref)
#         # prtrial = snu.get_pfromv(v=vref, **snsedict)
#         # vrtrial, prtrial = expand_vp_dolfunc(PrP, vc=vref, pc=prtrial)
#         # print 'pref', dolfin.norm(preff)
#         # print 'p', dolfin.norm(p)
#         # print 'p(v)', dolfin.norm(ptrial)
#         # print 'p(vref){0}\n'.format(dolfin.norm(prtrial))
#         elv.append(dolfin.errornorm(v, vreff))
#         elp.append(dolfin.errornorm(p, preff))
#         # elv.append(dolfin.norm(vdiff))
#         # elp.append(dolfin.norm(pdiff))
#         # cres = J*vp[:Nv]-fpbc
#         # mpires = (Mpfac.solve(cres.flatten())).reshape((cres.size, 1))
#         # ncres = np.sqrt(np.dot(cres.T, mpires))[0][0]
#         # ncres = np.sqrt(np.dot(cres.T, MP*cres))[0][0]
#         # routine from time_int_schemes seems buggy for CR or 'g not 0'
#         # ncres = comp_cont_error(v, fpbc, PrP.Q)
#         elc.append(ncres)
#     trange = np.linspace(0., tE, Nts+1)
#     samplvec = np.arange(1, len(trange), samplerate)
#     app_pverr(0.)
#     for t in trange[samplvec]:
#         app_pverr(t)
#     ev = np.array(elv)
#     ep = np.array(elp)
#     ec = np.array(elc)
#     trange = np.r_[trange[0], trange[samplerate]]
#     dtvec = trange[1:] - trange[:-1]
#
#     trapv = 0.5*(ev[:-1] + ev[1:])
#     errv = (dtvec*trapv).sum()
#
#     trapp = 0.5*(ep[:-1] + ep[1:])
#     errp = (dtvec*trapp).sum()
#
#     trapc = 0.5*(ec[:-1] + ec[1:])
#     resc = (dtvec*trapc).sum()
#
#     # print 'Nts = {0}, v_error = {1}, p_error = {2}, contres={3}'.\
#     #     format(Nts, errv, errp, resc)
#
#     errvl.append(errv)
#     errpl.append(errp)
#     rescl.append(resc)
#
# # print errvl
# # print errpl
# # print rescl
#
# # topgfplot = True
# # if topgfplot:
# #     ltpl = [errvl, errpl, rescl]
# #     for ltp in ltpl:
# #         for (i, Nts) in enumerate(Ntslist):
# #             print '({0}, {1})'.format(1./Nts, ltp[i])
#
# cpu.conv_plot(Ntslist, [errvl], logscale=2,
#               markerl=['o'], fignum=1, leglist=['velerror'])
# cpu.conv_plot(Ntslist, [rescl], logscale=2,
#               markerl=['o'], fignum=2, leglist=['cres'])
# cpu.conv_plot(Ntslist, [errpl], logscale=2,
#               markerl=['o'], fignum=3, leglist=['perror'])
