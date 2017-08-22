import os

import dolfin
import numpy as np
# from time_int_schemes import expand_vp_dolfunc, get_dtstr

import dolfin_navier_scipy.stokes_navier_utils as snu
# import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.dolfin_to_sparrays as dts

# import matlibplots.conv_plot_utils as cpu

import tdp_nse_schemes as tns
import helpers as hlp

dolfin.set_log_level(60)

samplerate = 1
plotplease = True

Nref = 3
N, Re, scheme, t0, tE = 3, 60, 'TH', 0., 1.
# Ntslist = [2**x for x in range(6, 11)]
Ntslist = [2**x for x in range(9, 11)]  # 2**x for x in range(5, 6)]
Ntsref = 2048
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


refcoeffs = hlp.getthecoeffs(N=Nref, Re=Re, scheme=scheme)

# get the ref trajectories
trange = np.linspace(t0, tE, Ntsref+1)

parastr = 'Re{0}N{1}scheme{5}Nts{2}t0{3}tE{4}'.\
    format(Re, Nref, Ntsref, t0, tE, scheme)

snsedict = dict(A=refcoeffs['A'], J=refcoeffs['J'], JT=refcoeffs['JT'],
                M=refcoeffs['M'], ppin=refcoeffs['ppin'], fv=refcoeffs['fv'],
                fp=refcoeffs['fp'], V=refcoeffs['V'], Q=refcoeffs['Q'],
                invinds=refcoeffs['invinds'], diribcs=refcoeffs['diribcs'],
                iniv=refcoeffs['iniv'], trange=trange,
                nu=refcoeffs['femp']['nu'],
                clearprvdata=False, paraviewoutput=True,
                nsects=10, addfullsweep=True,
                vel_pcrd_stps=1,
                data_prfx=datapathref + parastr,
                vfileprfx=plotspath+'refv_',
                pfileprfx=plotspath+'refp_',
                return_dictofpstrs=True, return_dictofvelstrs=True)

vdref, pdref = snu.solve_nse(**snsedict)


def compvperror(reffemp=None, vref=None, pref=None,
                curfemp=None, vcur=None, pcur=None):
    try:
        verf, perf = dts.expand_vp_dolfunc(vc=vref-vcur, pc=pref-pcur,
                                           zerodiribcs=True, **reffemp)
        verr = dolfin.norm(verf)
        perr = dolfin.norm(perf)
        # vreff, preff = dts.expand_vp_dolfunc(vc=vref, pc=pref, **reffemp)
        # vcurf, pcurf = dts.expand_vp_dolfunc(vc=vcur, pc=pcur, **curfemp)
        # verr = dolfin.norm(vreff - vcurf)
        # perr = dolfin.norm(preff - pcurf)
    except ValueError:  # obviously not the same FEM spaces
        vreff, preff = dts.expand_vp_dolfunc(vc=vref, pc=pref, **reffemp)
        vcurf, pcurf = dts.expand_vp_dolfunc(vc=vcur, pc=pcur, **curfemp)
        verr = dolfin.errornorm(vreff, vcurf)
        perr = dolfin.errornorm(preff, pcurf)
    return verr, perr


coeffs = hlp.getthecoeffs(N=N, Re=Re, scheme=scheme)

errvl = []
errpl = []

methdict = {'imexeuler': tns.halfexp_euler_nseind2,
            'projectn2': tns.projection2}

curmethnm = 'projectn2'
curmethod = methdict[curmethnm]
for Nts in Ntslist:
    parastr = 'Re{0}N{1}scheme{5}Nts{2}t0{3}tE{4}'.\
        format(Re, N, Nts, t0, tE, scheme)

    def getdatastr(t=None):
        return datapath + curmethnm + parastr + 't{0:.5f}'.format(t)

    vfile = plotspath + curmethnm + 'vels'
    pfile = plotspath + curmethnm + 'pres'
    plotit = hlp.getparaplotroutine(femp=coeffs['femp'], plotplease=plotplease,
                                    vfile=vfile, pfile=pfile)
    curttrange = np.linspace(t0, tE, Nts+1)
    curslvdct = coeffs
    curslvdct.update(trange=curttrange, get_datastr=getdatastr,
                     plotroutine=plotit, numoutputpts=100,
                     getconvfv=coeffs['getconvvec'])

    vdcur, pdcur = curmethod(**curslvdct)
    trangeidx = [0, np.int(np.floor(Nts/2)), -1]
    for trix in trangeidx:
        rsv = np.load(vdref[curttrange[trix]] + '.npy')
        csv = np.load(vdcur[curttrange[trix]] + '.npy')
        print('ref vs. curv at t={0}: {1}'.
              format(curttrange[trix], np.linalg.norm(rsv-csv)))

    elv = []
    elp = []

    for tcur in curttrange:
        vref = np.load(vdref[tcur] + '.npy')
        pref = np.load(pdref[tcur] + '.npy')
        vcur = np.load(vdcur[tcur] + '.npy')
        pcur = np.load(pdcur[tcur] + '.npy')
        verc, perc = compvperror(reffemp=refcoeffs['femp'],
                                 curfemp=coeffs['femp'],
                                 vref=vref, pref=pref, vcur=vcur, pcur=pcur)
        elv.append(verc)
        elp.append(perc)

    ev = np.array(elv)
    ep = np.array(elp)
    dtvec = curttrange[1:] - curttrange[:-1]

    trapv = 0.5*(ev[:-1] + ev[1:])
    errv = (dtvec*trapv).sum()

    trapp = 0.5*(ep[:-1] + ep[1:])
    errp = (dtvec*trapp).sum()

    errvl.append(errv)
    errpl.append(errp)

print('integrated errors: Nts: v_e ; p_e')
for k, Nts in enumerate(Ntslist):
    errv, errp = errvl[k], errpl[k]
    print('{0}: {1}; {2}'.format(Nts, errv, errp))

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
