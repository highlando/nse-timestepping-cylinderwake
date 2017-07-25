import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla


def halfexp_euler_nseind2(Mc=None, MP=None, Ac=None, BTc=None, Bc=None,
                          fvbc=None, fpbc=None, fconv=None, ppin=None,
                          getconvfv=None,
                          Nts=100, t0=0., tE=1., numoutputpts=100,
                          linatol=0,
                          get_datastr=None, plotroutine=None,
                          vp_init=None):
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
    # | 1/dt*M  -B.T |   | q+|     | 1/dt*M*qc - K(qc) + fc |
    # |              | * |   |  =  |                        |
    # |    B         |   | pc|     | g                      |
    #
    #

    dt = (t0 - tE)/Nts
    Nv = Ac.shape[0]

    tcur = t0

    MFac = dt
    CFac = 1  # /dt
    # PFac = -1  # -1 for symmetry (if CFac==1)
    PFacI = -1./dt

    cdatstr = get_datastr(t=t0)

    try:
        np.load(cdatstr + '.npy')
        print 'loaded data from ', cdatstr, ' ...'
    except IOError:
        np.save(cdatstr, vp_init)
        print 'saving to ', cdatstr, ' ...'

    plotroutine(vp_init, t=tcur)
    # v, p = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
    # TsP.UpFiles.u_file << v, tcur
    # TsP.UpFiles.p_file << p, tcur
    Bcc, BTcc, MPc, fpbcc, vp_init, Npc = pinthep(Bc, BTc, MP, fpbc,
                                                  vp_init, ppin)

    IterAv = MFac*sps.hstack([1.0/dt*Mc + Ac, PFacI*(-1)*BTcc])
    IterAp = CFac*sps.hstack([Bcc, sps.csr_matrix((Npc, Npc))])
    IterA = sps.vstack([IterAv, IterAp])
    if linatol == 0:
        IterAfac = spsla.factorized(IterA)

    vp_old = vp_init
    vp_old = np.vstack([vp_init[:Nv], 1./PFacI*vp_init[Nv:]])
    vp_oldold = vp_old
    ContiRes, VelEr, PEr, TolCorL = [], [], [], []

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

    for etap in range(1, numoutputpts + 1):
        for i in range(Nts / numoutputpts):
            cdatstr = get_datastr(t=tcur+dt)
            try:
                vp_next = np.load(cdatstr + '.npy')
                print 'loaded data from ', cdatstr, ' ...'
                vp_next = np.vstack([vp_next[:Nv], 1./PFacI*vp_next[Nv:]])
                vp_oldold = vp_old
                vp_old = vp_next
                if tcur == dt+dt:
                    iniiterfac = 1  # fac only in the first Krylov Call
            except IOError:
                print 'computing data for ', cdatstr, ' ...'
                curconfv = getconvfv(v)
                # ConV = dts.get_convvec(u0_dolfun=v, V=PrP.V)
                # CurFv = dts.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

                Iterrhs = np.vstack([MFac*1.0/dt*Mc*vp_old[:Nv, ],
                                     np.zeros((Npc, 1))]) +\
                    np.vstack([MFac*(fvbc + CurFv - ConV[PrP.invinds, ]),
                               CFac*fpbcc])

                if TsP.linatol == 0:
                    # ,vp_old,tol=TsP.linatol)
                    vp_new = IterAfac(Iterrhs.flatten())
                    # vp_new = spsla.spsolve(IterA, Iterrhs)
                    vp_old = np.atleast_2d(vp_new).T
                    TolCor = 0

                else:
                    if inikryupd and tcur == t0:
                        print '\n1st step direct solve to initialize krylov\n'
                        vp_new = spsla.spsolve(IterA, Iterrhs)
                        vp_old = np.atleast_2d(vp_new).T
                        TolCor = 0
                        inikryupd = False  # only once !!
                    else:
                        if TsP.TolCorB:
                            NormRhsInd2 = \
                                np.sqrt(ind2_ip(Iterrhs, Iterrhs))[0][0]
                            TolCor = 1.0 / np.max([NormRhsInd2, 1])
                        else:
                            TolCor = 1.0

                        curls = krypy.linsys.LinearSystem(IterA, Iterrhs,
                                                          M=MInv)

                        tstart = time.time()

                        # extrapolating the initial value
                        upv = (vp_old - vp_oldold)

                        ret = krypy.linsys.\
                            RestartedGmres(curls, x0=vp_old + upv,
                                           tol=TolCor*TsP.linatol,
                                           maxiter=iniiterfac*TsP.MaxIter,
                                           max_restarts=100)

                        # ret = krypy.linsys.\
                        #     Minres(curls, maxiter=20*TsP.MaxIter,
                        #            x0=vp_old + upv, tol=TolCor*TsP.linatol)
                        tend = time.time()
                        vp_oldold = vp_old
                        vp_old = ret.xk

                        print ('Needed {0} of max {4}*{1} iterations: ' +
                               'final relres = {2}\n TolCor was {3}').\
                            format(len(ret.resnorms), TsP.MaxIter,
                                   ret.resnorms[-1], TolCor, iniiterfac)
                        print 'Elapsed time {0}'.format(tend - tstart)
                        iniiterfac = 1  # fac only in the first Krylov Call

                np.save(cdatstr, np.vstack([vp_old[:Nv],
                                            PFacI*vp_old[Nv:]]))

            vc = vp_old[:Nv, ]
            print 'Norm of current v: ', np.linalg.norm(vc)
            pc = PFacI*vp_old[Nv:, ]

            v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc)

            tcur += dt

            # the errors
            vCur, pCur = PrP.v, PrP.p
            try:
                vCur.t = tcur
                pCur.t = tcur - dt

                ContiRes.append(comp_cont_error(v, fpbc, PrP.Q))
                VelEr.append(errornorm(vCur, v))
                PEr.append(errornorm(pCur, p))
                TolCorL.append(TolCor)
            except AttributeError:
                ContiRes.append(0)
                VelEr.append(0)
                PEr.append(0)
                TolCorL.append(0)

        print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts)

        if TsP.ParaviewOutput:
            TsP.UpFiles.u_file << v, tcur
            TsP.UpFiles.p_file << p, tcur

    TsP.Residuals.ContiRes.append(ContiRes)
    TsP.Residuals.VelEr.append(VelEr)
    TsP.Residuals.PEr.append(PEr)
    TsP.TolCor.append(TolCorL)

    return


def pinthep(B, BT, M, fp, vp_init, pdof):
    """remove dofs of div and grad to pin the pressure

    """
    (NP, NV) = B.shape
    if pdof is None:
        return B, BT, M, fp, vp_init, NP
    elif pdof == 0:
        vpi = np.vstack([vp_init[:NV, :], vp_init[NV+1:, :]])
        return (B[1:, :], BT[:, 1:], M[1:, :][:, 1:], fp[1:, :],
                vpi, NP - 1)
    elif pdof == -1:
        return (B[:-1, :], BT[:, :-1], M[:-1, :][:, :-1],
                fp[:-1, :], vp_init[:-1, :], NP - 1)
    else:
        raise NotImplementedError()
