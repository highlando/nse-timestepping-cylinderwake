import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts


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
