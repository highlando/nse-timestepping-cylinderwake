import os

import dolfin
import numpy as np
import scipy.sparse.linalg as spsla
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

Nref = 2
N, Re, scheme, t0, tE = 2, 60, 'TH', 0., 1.
# Ntslist = [2**x for x in range(6, 11)]
Ntslist = [2**x for x in range(8, 12)]
# Ntslist = [2**x for x in range(8, 11)]
# Ntslist = [2**x for x in range(9, 11)]
Ntsref = 2048
tol = 2**(-16)

curmethnm = 'projectn2'
curmethnm = 'SIMPLE'
linatollist = [0, 1e-4, 1e-5, 1e-6]  # , 1e-6, 1e-7]

curmethnm = 'imexeuler'
linatollist = [0, 1e-6, 1e-7, 1e-8]

methdict = {'imexeuler': tns.halfexp_euler_nseind2,
            'projectn2': tns.projection2,
            'SIMPLE': tns.SIMPLE}

curmethod = methdict[curmethnm]


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

vel, pel, crl, mrl = [], [], [], []
for linatol in linatollist:
    errvl, errpl, crrsl, mrrsl = [], [], [], []

    for Nts in Ntslist:
        parastr = 'Re{0}N{1}scheme{5}Nts{2}t0{3}tE{4}linatol{6:.2e}'.\
            format(Re, N, Nts, t0, tE, scheme, linatol)
        # parastr = 'Re{0}N{1}scheme{5}Nts{2}t0{3}tE{4}'.\
        #     format(Re, N, Nts, t0, tE, scheme)

        def getdatastr(t=None):
            return datapath + curmethnm + parastr + 't{0:.5f}'.format(t)

        vfile = plotspath + curmethnm + 'vels'
        pfile = plotspath + curmethnm + 'pres'
        plotit = hlp.getparaplotroutine(femp=coeffs['femp'],
                                        plotplease=plotplease,
                                        vfile=vfile, pfile=pfile)
        curttrange = np.linspace(t0, tE, Nts+1)
        curslvdct = coeffs
        curslvdct.update(trange=curttrange, get_datastr=getdatastr,
                         plotroutine=plotit, numoutputpts=100,
                         linatol=linatol,
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
        crs = []
        mrs = []
        MP, J, fp, A = coeffs['MP'], coeffs['J'], coeffs['fp'], coeffs['A']
        M, getconvfv, fv = coeffs['M'], coeffs['getconvfv'], coeffs['fv']
        mpfac = spsla.factorized(MP)
        mfac = spsla.factorized(M)
        v_old = None

        for tk, tcur in enumerate(curttrange):
            vref = np.load(vdref[tcur] + '.npy')
            pref = np.load(pdref[tcur] + '.npy')
            vcur = np.load(vdcur[tcur] + '.npy')
            pcur = np.load(pdcur[tcur] + '.npy')
            verc, perc = compvperror(reffemp=refcoeffs['femp'],
                                     curfemp=coeffs['femp'], vref=vref,
                                     pref=pref, vcur=vcur, pcur=pcur)

            if tk > 0:
                dt = tcur - curttrange[tk-1]
                curconfv = getconvfv(v_old)
                meqrhs = 1.0/dt*M*v_old + fv - curconfv + J.T*pcur
                cmrsv = 1.0/dt*M*vcur + A*vcur - meqrhs
                nmqrhs = np.dot(meqrhs.T, mfac(meqrhs))
                mrs.append(np.sqrt(np.dot(cmrsv.T, mfac(cmrsv))/nmqrhs))

                crscv = J*vcur - fp
                crs.append(np.sqrt(np.dot(crscv.T, mpfac(crscv))/nmqrhs))
            else:
                mrs.append(0.)
                crs.append(0.)
            v_old = vcur
            elv.append(verc)
            elp.append(perc)

        ev = np.array(elv)
        ep = np.array(elp)
        cr = np.array(crs)

        mr = np.array(mrs)
        dtvec = curttrange[1:] - curttrange[:-1]
        trapv = 0.5*(ev[:-1] + ev[1:])
        errv = (dtvec*trapv).sum()
        trapp = 0.5*(ep[:-1] + ep[1:])
        errp = (dtvec*trapp).sum()
        trapcr = 0.5*(cr[:-1] + cr[1:])
        crrs = (dtvec*trapcr).sum()
        trapmr = 0.5*(mr[:-1] + mr[1:])
        mrrs = (dtvec*trapmr).sum()

        errvl.append(errv)
        errpl.append(errp)
        crrsl.append(crrs)
        mrrsl.append(mrrs)

    print('integrated errors: Nts: v_e ; p_e ; cres ; mres')
    for k, Nts in enumerate(Ntslist):
        errv, errp, crr, mrr = errvl[k], errpl[k], crrsl[k], mrrsl[k]
        print('{0}: {1}; {2}; {3}; {4}'.format(Nts, errv, errp, crr, mrr))
    vel.append(errvl)
    pel.append(errpl)
    crl.append(crrsl)
    mrl.append(mrrsl)

import matlibplots.conv_plot_utils as cpu
tkzprfx = 'tikzs/' + curmethnm
markers = ['x', 'o', 'v', '^']
leglist = ['{0:.2e}'.format(ltol) for ltol in linatollist]
abscissa = 1./np.array(Ntslist)
markerl = markers[:len(vel)]
pltdct = dict(logscale=True, markerl=markerl, logbase=10,
              leglist=leglist, legloc=4)
cpu.conv_plot(abscissa, vel, fignum=1, tikzfile=tkzprfx+'velerrs.tikz',
              title='vel apprx', **pltdct)
cpu.conv_plot(abscissa, pel, fignum=2, tikzfile=tkzprfx+'pelerrs.tikz',
              title='pres apprx', **pltdct)

# don't need the residuals for the exact solves
crpcrl = crl[1:]
crpmrl = mrl[1:]
markerl = markers[1:len(crl)+1]
leglistc = leglist[1:len(crl)+1]
pltdct.update(dict(markerl=markerl, leglist=leglistc))
cpu.conv_plot(abscissa, crpcrl, fignum=3, tikzfile=tkzprfx+'cres.tikz',
              title='conti residual', **pltdct)
cpu.conv_plot(abscissa, crpmrl, fignum=4, tikzfile=tkzprfx+'meqres.tikz',
              title='momeq residual', **pltdct)
