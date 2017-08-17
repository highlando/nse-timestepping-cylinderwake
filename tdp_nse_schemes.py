import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla


def halfexp_euler_nseind2(M=None, MP=None, A=None, JT=None, J=None,
                          fv=None, fp=None, ppin=None,
                          getconvfv=None,
                          Nts=1000, t0=0., tE=1.,
                          trange=None,
                          numoutputpts=10,
                          linatol=0,
                          get_datastr=None, plotroutine=None,
                          verbose=True,
                          iniv=None, inip=None,
                          **kwargs):
    """halfexplicit euler for the NSE in index 2 formulation

    Parameters
    ----------
    numoutputpts : int, optional
        number of points at which the computed quantities are logged
    getconvfv : f(v), callable
        returns the convection term to be added to the right hand side
    """
    #
    #
    # Basic Eqn:
    #
    # | 1/dt*M  -J.T |   | q+|     | 1/dt*M*qc - K(qc) + fc |
    # |              | * |   |  =  |                        |
    # |    J         |   | pc|     | g                      |
    #
    #

    dt = (tE - t0)/Nts
    trange = np.linspace(t0, tE, Nts+1)
    Nv = A.shape[0]

    tcur = t0

    MFac = dt
    CFac = 1  # /dt
    PFacI = 1.  # -1./dt

    dictofvstrs, dictofpstrs = {}, {}

    cdatstr = get_datastr(t=t0)

    # try:
    #     np.load(cdatstr + '.npy')
    #     print 'loaded data from ', cdatstr, ' ...'
    # except IOError:
    inivp = np.vstack([iniv, inip])
    np.save(cdatstr + '_v', iniv)
    np.save(cdatstr + '_p', inip)
    dictofvstrs.update({t0: cdatstr + '_v'})
    dictofpstrs.update({t0: cdatstr + '_p'})

    # print 'saving to ', cdatstr, ' ...'

    plotroutine(inivp, t=tcur)
    # v, p = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
    # TsP.UpFiles.u_file << v, tcur
    # TsP.UpFiles.p_file << p, tcur
    J, JT, MP, fp, vp_init, Npc = pinthep(J, JT, MP, fp, inivp, ppin)

    IterAv = MFac*sps.hstack([1.0/dt*M + A, PFacI*(-1)*JT])
    IterAp = CFac*sps.hstack([J, sps.csr_matrix((Npc, Npc))])
    IterA = sps.vstack([IterAv, IterAp])
    if linatol == 0:
        IterAfac = spsla.factorized(IterA)

    vp_old = vp_init
    # vp_oldold = vp_old
    # TolCorL = []

    # Mvp = sps.csr_matrix(sps.block_diag((Mc, MPc)))
    # Mvp = sps.eye(Mc.shape[0] + MPc.shape[0])
    # Mvp = None

    # M matrix for the minres routine
    # M accounts for the FEM discretization

    # Mcfac = spsla.splu(Mc)
    # MPcfac = spsla.splu(MPc)

    # def _MInv(vp):
    #     # v, p = vp[:Nv, ], vp[Nv:, ]
    #     # lsv = krypy.linsys.LinearSystem(Mc, v, self_adjoint=True)
    #     # lsp = krypy.linsys.LinearSystem(MPc, p, self_adjoint=True)
    #     # Mv = (krypy.linsys.Cg(lsv, tol=1e-14)).xk
    #     # Mp = (krypy.linsys.Cg(lsp, tol=1e-14)).xk
    #     v, p = vp[:Nv, ], vp[Nv:, ]
    #     Mv = np.atleast_2d(Mcfac.solve(v.flatten())).T
    #     Mp = np.atleast_2d(MPcfac.solve(p.flatten())).T
    #     return np.vstack([Mv, Mp])

    # MInv = spsla.LinearOperator(
    #     (Nv + Npc,
    #      Nv + Npc),
    #     matvec=_MInv,
    #     dtype=np.float32)

    # def ind2_ip(vp1, vp2):
    #     """

    #     for applying the fem inner product
    #     """
    #     v1, v2 = vp1[:Nv, ], vp2[:Nv, ]
    #     p1, p2 = vp1[Nv:, ], vp2[Nv:, ]
    #     return mass_fem_ip(v1, v2, Mcfac) + mass_fem_ip(p1, p2, MPcfac)

    # inikryupd = TsP.inikryupd
    # iniiterfac = TsP.iniiterfac  # the first krylov step needs more maxiter

    # for etap in range(1, numoutputpts + 1):
    #     for i in range(np.int(Nts/numoutputpts)):
    if verbose:
        print('IMEX Euler on [{0}, {1}]'.format(trange[0], trange[-1]))
    for tk, tcur in enumerate(trange[1:]):
        cts = tcur - trange[tk]
        if not np.allclose(cts, dt) and not linatol == 0:
            raise UserWarning('trange not equispaced -- cannot prefac A')
        cdatstr = get_datastr(t=tcur)
        try:
            vp_next = np.load(cdatstr + '.npy')
            print('loaded data from ', cdatstr, ' ...')
            vp_next = np.vstack([vp_next[:Nv], vp_next[Nv:]])
            # vp_oldold = vp_old
            vp_old = vp_next
            # if tcur == dt+dt:
            #     iniiterfac = 1  # fac only in the first Krylov Call
        except IOError:
            print('computing data for ', cdatstr, ' ...')
            v_old = vp_old[:Nv]
            curconfv = getconvfv(v_old)
            # ConV = dts.get_convvec(u0_dolfun=v, V=PrP.V)
            # CurFv = dts.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

            Iterrhs = np.vstack([MFac*1.0/dt*M*v_old,
                                 np.zeros((Npc, 1))]) +\
                np.vstack([MFac*(fv - curconfv), CFac*fp])

            if linatol == 0:  # direct solve!
                # ,vp_old,tol=TsP.linatol)
                vp_new = IterAfac(Iterrhs.flatten())
                # vp_new = spsla.spsolve(IterA, Iterrhs)
                vp_old = np.atleast_2d(vp_new).T
                # TolCor = 0

            # else:
            #     if inikryupd and tcur == t0:
            #         print '\n1st step direct solve to init krylov\n'
            #         vp_new = spsla.spsolve(IterA, Iterrhs)
            #         vp_old = np.atleast_2d(vp_new).T
            #         TolCor = 0
            #         inikryupd = False  # only once !!
            #     else:
            #         if TsP.TolCorB:
            #             NormRhsInd2 = \
            #                 np.sqrt(ind2_ip(Iterrhs, Iterrhs))[0][0]
            #             TolCor = 1.0 / np.max([NormRhsInd2, 1])
            #         else:
            #             TolCor = 1.0

            #         curls = krypy.linsys.LinearSystem(IterA, Iterrhs,
            #                                           M=MInv)

            #         tstart = time.time()

            #         # extrapolating the initial value
            #         upv = (vp_old - vp_oldold)

            #         ret = krypy.linsys.\
            #             RestartedGmres(curls, x0=vp_old + upv,
            #                            tol=TolCor*TsP.linatol,
            #                            maxiter=iniiterfac*TsP.MaxIter,
            #                            max_restarts=100)

            #         # ret = krypy.linsys.\
            #         #     Minres(curls, maxiter=20*TsP.MaxIter,
            #         #            x0=vp_old + upv, tol=TolCor*TsP.linatol)
            #         tend = time.time()
            #         vp_oldold = vp_old
            #         vp_old = ret.xk

            #         print ('Needed {0} of max {4}*{1} iterations: ' +
            #                'final relres = {2}\n TolCor was {3}').\
            #             format(len(ret.resnorms), TsP.MaxIter,
            #                    ret.resnorms[-1], TolCor, iniiterfac)
            #         print 'Elapsed time {0}'.format(tend - tstart)
            #         iniiterfac = 1  # fac only in the first Krylov Call

            np.save(cdatstr + '_v', vp_old[:Nv])
            np.save(cdatstr + '_p', PFacI*vp_old[Nv:])
            dictofvstrs.update({tcur: cdatstr + '_v'})
            dictofpstrs.update({tcur: cdatstr + '_p'})

            # TolCorL.append(TolCor)

        if np.mod(tk+1, np.int(np.floor(Nts/numoutputpts))) == 0:
            plotroutine(vp_old, t=tcur)
            if verbose:
                print('{0}/{1} time steps completed'.format(tk, Nts))

    return dictofvstrs, dictofpstrs


def pinthep(J, JT, M, fp, vp_init, pdof):
    """remove dofs of div and grad to pin the pressure

    """
    (NP, NV) = J.shape
    if pdof is None:
        return J, JT, M, fp, vp_init, NP
    elif pdof == 0:
        vpi = np.vstack([vp_init[:NV, :], vp_init[NV+1:, :]])
        return (J[1:, :], JT[:, 1:], M[1:, :][:, 1:], fp[1:, :],
                vpi, NP - 1)
    elif pdof == -1:
        return (J[:-1, :], JT[:, :-1], M[:-1, :][:, :-1],
                fp[:-1, :], vp_init[:-1, :], NP - 1)
    else:
        raise NotImplementedError()
