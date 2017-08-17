import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps
import sadptprj_riclyap_adi.lin_alg_utils as lau


def getthecoeffs(N=None, Re=None, scheme='CR',
                 inivp='Stokes', inifemp=None):
    femp, stokesmatsc, rhsmomcont \
        = dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                            scheme=scheme, mergerhs=True)
    fp = rhsmomcont['fp']
    fv = rhsmomcont['fv']
    # stokesmatsc.update(rhsmomcont)

    J, MP = stokesmatsc['J'], stokesmatsc['MP']
    NV, NP = J.T.shape
    # Nv = J.shape[1]
    # Mpfac = spsla.splu(MP)

    if inivp is 'Stokes':
        inivp = lau.solve_sadpnt_smw(amat=stokesmatsc['A'], jmat=J,
                                     jmatT=-J.T, rhsv=fv, rhsp=fp)
        iniv = inivp[:NV]
        inip = snu.get_pfromv(v=iniv, V=femp['V'], M=stokesmatsc['M'],
                              A=stokesmatsc['A'], J=J, fv=fv, fp=fp,
                              diribcs=femp['diribcs'], invinds=femp['invinds'])
    else:
        inv, inp = dts.expand_vp_dolfunc(vp=inivp, **inifemp)
        # interpolate on new mesh and extract the invinds

    getconvvec = getconvvecfun(**femp)
    return dict(A=stokesmatsc['A'], M=stokesmatsc['M'], J=J, JT=J.T, MP=MP,
                fp=fp, fv=fv, getconvvec=getconvvec,
                iniv=iniv, inip=inip,
                V=femp['V'], Q=femp['Q'], invinds=femp['invinds'],
                diribcs=femp['diribcs'], ppin=femp['ppin'], femp=femp)


def getconvvecfun(V=None, diribcs=None, invinds=None, **kwargs):
    def getconvvec(vvec):
        return dts.get_convvec(u0_vec=vvec, V=V,
                               diribcs=diribcs, invinds=invinds)
    return getconvvec


def getparaplotroutine(femp=None, vfile=None, pfile=None, plotplease=True):

    vfile = dolfin.File(vfile + '.pvd')
    pfile = dolfin.File(pfile + '.pvd')

    def plotit(vp, t):
        if plotplease:
            v, p = dts.expand_vp_dolfunc(vp=vp, **femp)
            v.rename('v', 'velocity')
            p.rename('p', 'pressure')
            vfile << v, t
            pfile << p, t
        else:
            return
    return plotit
