import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# import time


def SIMPLE(M=None, MP=None, A=None, JT=None, J=None,
           fv=None, fp=None, ppin=None,
           iniv=None, inip=None,
           getconvfv=None,
           Nts=None, t0=None, tE=None, trange=None,
           numoutputpts=10,
           linatol=0,
           get_datastr=None, plotroutine=None, verbose=False,
           **kwargs):
    """SIMPLE scheme for time integration of NSE

    Basic Eqn:

           (1/dt*M + A)*tv+ = 1/dt*M*vc + J.T*pc - K(qc) + fc
    J*(1/dt*M+A).-1*J.T*dp+ = -J*tv+ + g+
                         v+ = tv+ + (1/dt*M+A).-1*J.T*dp+
                         p+ = pc + dp+

    Parameters
    ----------
    numoutputpts : int, optional
        number of points at which the computed quantities are logged
    getconvfv : f(v), callable
        returns the convection term to be added to the right hand side
    """

    t0, tE, Nts, dt = _setuptdisc(trange=trange, t0=t0, tE=tE, Nts=Nts)

    Np, Nv = J.shape[0], J.shape[1]

    tcur = t0

    dictofvstrs, dictofpstrs = {}, {}

    cdatstr = get_datastr(t=t0)

    inivp = np.vstack([iniv, inip])
    np.save(cdatstr + '_v', iniv)
    np.save(cdatstr + '_p', inip)
    dictofvstrs.update({t0: cdatstr + '_v'})
    dictofpstrs.update({t0: cdatstr + '_p'})

    plotroutine(inivp, t=tcur)
    J, JT, MP, fp, vp_init, Npc = pinthep(J, JT, MP, fp, inivp, ppin)

    momeqmat = 1.0/dt*M + A
    if linatol == 0:
        momeqfac = spsla.factorized(momeqmat)
        projmat = sps.vstack([sps.hstack([1./dt*M+A, -JT]),
                              sps.hstack([J, sps.csr_matrix((Np, Np))])])
        prjmatfac = spsla.factorized(projmat)

    else:
        import krypy
        momeqfac = spsla.factorized(momeqmat)
        mfac = spsla.factorized(M)
        mpfac = spsla.factorized(MP)

        def _appMomeqinv(v):
            MmqinvV = momeqfac(v)
            return MmqinvV.reshape((v.size, 1))

        def _invBMmqBv(v):
            bv = JT*v
            minbv = np.atleast_2d(momeqfac(bv.flatten())).T
            return J*minbv

        def _invMP(p):
            return np.atleast_2d(mpfac(p.flatten())).T

        def _invM(v):
            return np.atleast_2d(mfac(v.flatten())).T

        BMmqpmoBT = spsla.LinearOperator((Np, Np), matvec=_invBMmqBv,
                                         dtype=np.float32)
        invMP = spsla.LinearOperator((Np, Np), matvec=_invMP, dtype=np.float32)
        invM = spsla.LinearOperator((Nv, Nv), matvec=_invM, dtype=np.float32)

    v_old = iniv
    p_old = inip

    if verbose:
        print('SIMPLE on [{0}, {1}]'.format(trange[0], trange[-1]))
    for tk, tcur in enumerate(trange[1:]):
        cdatstr = get_datastr(t=tcur)
        try:
            v_next = np.load(cdatstr + '_v.npy')
            p_next = np.load(cdatstr + '_p.npy')
            loadorcomped = 'loaded'
            if verbose:
                print('loaded data from ', cdatstr, ' ...')
        except IOError:
            loadorcomped = 'computed'
            if verbose:
                print('computing data for ', cdatstr, ' ...')
            curconfv = getconvfv(v_old)
            meqrhs = 1.0/dt*M*v_old + fv - curconfv + J.T*p_old

            if linatol == 0:  # direct solve!
                tvn = momeqfac(meqrhs.flatten())
                tvn = np.atleast_2d(tvn).T
                vndpn = prjmatfac(np.vstack([1./dt*M*tvn+A*tvn, fp]))
                v_next = vndpn[:-Np].reshape((tvn.size, 1))
                dpn = vndpn[-Np:].reshape((Np, 1))
                p_next = p_old + dpn
                # print('res(vn,pn): {0}'.format(0)

            else:
                mls = krypy.linsys.LinearSystem(momeqmat, meqrhs, M=invM)
                tvn = krypy.linsys.Cg(mls, tol=.5*linatol).xk
                pperhs = J*tvn - fp
                pls = krypy.linsys.LinearSystem(BMmqpmoBT, pperhs, M=invMP)
                dpn = krypy.linsys.Cg(pls, tol=.5*linatol).xk
                v_next = tvn + momeqfac(JT*dpn)
                p_next = p_old + dpn

            v_old = v_next
            p_old = p_next
            np.save(cdatstr + '_v', v_old)
            np.save(cdatstr + '_p', p_old)

        dictofvstrs.update({tcur: cdatstr + '_v'})
        dictofpstrs.update({tcur: cdatstr + '_p'})

        if np.mod(tk+1, np.int(np.floor(Nts/numoutputpts))) == 0:
            plotroutine(np.vstack([v_old, p_old]), t=tcur)
            print('{0}/{1} time steps {2}'.format(tk, Nts, loadorcomped))

    return dictofvstrs, dictofpstrs


def projection2(M=None, MP=None, A=None, JT=None, J=None,
                fv=None, fp=None, ppin=None,
                getconvfv=None,
                Nts=None, t0=None, tE=None,
                trange=None,
                numoutputpts=10,
                linatol=0,
                get_datastr=None, plotroutine=None,
                verbose=False,
                iniv=None, inip=None,
                **kwargs):
    """projection2 method for time integration of NSE

    Basic Eqn:

    (1/dt*M + A)*tv+ = 1/dt*M*vc + J.T*pc - K(qc) + fc
    -J*M.-1*J.T*phi+ = J*tv+ - g+
                  v+ = tv+ + M.-1*J.T*phi+
                  p+ = pc + 0.5*dt*phi+

    Parameters
    ----------
    numoutputpts : int, optional
        number of points at which the computed quantities are logged
    getconvfv : f(v), callable
        returns the convection term to be added to the right hand side
    """

    t0, tE, Nts, dt = _setuptdisc(trange=trange, t0=t0, tE=tE, Nts=Nts)

    Np = J.shape[0]

    tcur = t0

    dictofvstrs, dictofpstrs = {}, {}

    cdatstr = get_datastr(t=t0)

    inivp = np.vstack([iniv, inip])
    np.save(cdatstr + '_v', iniv)
    np.save(cdatstr + '_p', inip)
    dictofvstrs.update({t0: cdatstr + '_v'})
    dictofpstrs.update({t0: cdatstr + '_p'})

    plotroutine(inivp, t=tcur)
    J, JT, MP, fp, vp_init, Npc = pinthep(J, JT, MP, fp, inivp, ppin)

    momeqmat = 1.0/dt*M + A
    if linatol == 0:
        momeqfac = spsla.factorized(momeqmat)

    # TODO: go back to PPE+cg
    projmat = sps.vstack([sps.hstack([2./dt*M, -2./dt*JT]),
                          sps.hstack([J, sps.csr_matrix((Np, Np))])])
    prjmatfac = spsla.factorized(projmat)

    import krypy
    mfac = spsla.factorized(M)
    mpfac = spsla.factorized(MP)

    def _appMinv(v):
        MinvV = mfac(v)
        return MinvV.reshape((v.size, 1))

    def _invBMBv(v):
        bv = JT*v
        minbv = np.atleast_2d(mfac(bv.flatten())).T
        return J*minbv

    def _invMP(p):
        return np.atleast_2d(mpfac(p.flatten())).T

    BMpmoBT = spsla.LinearOperator((Np, Np), matvec=_invBMBv, dtype=np.float32)
    invMP = spsla.LinearOperator((Np, Np), matvec=_invMP, dtype=np.float32)

    v_old = iniv
    p_old = inip

    if verbose:
        print('Projection2 on [{0}, {1}]'.format(trange[0], trange[-1]))
    for tk, tcur in enumerate(trange[1:]):
        cdatstr = get_datastr(t=tcur)
        try:
            v_next = np.load(cdatstr + '_v.npy')
            p_next = np.load(cdatstr + '_p.npy')
            loadorcomped = 'loaded'
            if verbose:
                print('loaded data from ', cdatstr, ' ...')
        except IOError:
            loadorcomped = 'computed'
            if verbose:
                print('computing data for ', cdatstr, ' ...')
            curconfv = getconvfv(v_old)
            meqrhs = 1.0/dt*M*v_old + fv - curconfv + J.T*p_old

            if linatol == 0:  # direct solve!
                tvn = momeqfac(meqrhs.flatten())
                tvn = np.atleast_2d(tvn).T
                vnphn = prjmatfac(np.vstack([2./dt*M*tvn, fp]))
                v_next = vnphn[:-Np].reshape((tvn.size, 1))
                phin = vnphn[-Np:].reshape((Np, 1))
                p_next = p_old + phin

            else:
                pperhs = -J*tvn + fp
                cls = krypy.linsys.LinearSystem(BMpmoBT, -pperhs, M=invMP)
                phin = krypy.linsys.Cg(cls).xk
                v_next = tvn + _appMinv(JT*phin)
                p_next = p_old + 2./dt*phin

                # TODO: go back to PPE+cg

            # print('PPE+cg vs. sadpt solve; dvn: {0}; dphin {1}'.
            #       format(np.linalg.norm(v_next-vn),
            #              np.linalg.norm(phin-phn)))
            v_old = v_next
            p_old = p_next
            np.save(cdatstr + '_v', v_old)
            np.save(cdatstr + '_p', p_old)

        dictofvstrs.update({tcur: cdatstr + '_v'})
        dictofpstrs.update({tcur: cdatstr + '_p'})

        if np.mod(tk+1, np.int(np.floor(Nts/numoutputpts))) == 0:
            plotroutine(np.vstack([v_old, p_old]), t=tcur)
            print('{0}/{1} time steps {2}'.format(tk, Nts, loadorcomped))

    return dictofvstrs, dictofpstrs


def halfexp_euler_nseind2(M=None, MP=None, A=None, JT=None, J=None,
                          fv=None, fp=None, ppin=None,
                          getconvfv=None,
                          Nts=None, t0=None, tE=None,
                          trange=None,
                          numoutputpts=10,
                          linatol=0,
                          get_datastr=None, plotroutine=None,
                          verbose=False,
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
    # | 1/dt*M + A  -J.T |   | q+|     | 1/dt*M*qc - K(qc) + fc |
    # |                  | * |   |  =  |                        |
    # |    J             |   | pc|     | g                      |
    #
    #

    try:
        t0, tE, Nts = trange[0], trange[-1], trange.size-1
    except TypeError:
        trange = np.linspace(t0, tE, Nts+1)

    dt = (tE - t0)/Nts
    dtvec = trange[1:] - trange[:-1]
    if not np.allclose(np.linalg.norm(dtvec)/np.sqrt(dtvec.size), dt):
        raise UserWarning('trange not equispaced -- cannot prefac A')

    Nv = A.shape[0]

    tcur = t0

    MFac = dt
    CFac = -1.  # /dt
    PFacI = 1./dt  # -1./dt

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
    vp_old = vp_init

    if linatol == 0:
        IterAfac = spsla.factorized(IterA)
    else:
        import krypy
        # TODO: make this an argument
        inikryupd = True
        # iniiterfac = 4
        # MaxIter = 600
        vp_oldold = vp_old
        Mfac = spsla.splu(M)
        MPfac = spsla.splu(MP)

        def _MInv(vp):
            v, p = vp[:Nv, ], vp[Nv:, ]
            Mv = np.atleast_2d(Mfac.solve(v.flatten())).T
            Mp = np.atleast_2d(MPfac.solve(p.flatten())).T
            return np.vstack([Mv, Mp])

        MInv = spsla.LinearOperator((Nv + Npc, Nv + Npc),
                                    matvec=_MInv, dtype=np.float32)

    # Mvp = sps.csr_matrix(sps.block_diag((Mc, MPc)))
    # Mvp = sps.eye(Mc.shape[0] + MPc.shape[0])
    # Mvp = None
    # M matrix for the minres routine
    # M accounts for the FEM discretization
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
    print('IMEX Euler on [{0}, {1}]'.format(trange[0], trange[-1]))
    for tk, tcur in enumerate(trange[1:]):
        cdatstr = get_datastr(t=tcur)
        try:
            v_next = np.load(cdatstr + '_v.npy')
            p_next = np.load(cdatstr + '_p.npy')
            if verbose:
                print('loaded data from ', cdatstr, ' ...')
            vp_next = np.vstack([v_next, p_next])
            # vp_oldold = vp_old
            vp_old = vp_next
            loadorcomped = 'loaded'
            # if tcur == dt+dt:
            #     iniiterfac = 1  # fac only in the first Krylov Call
        except IOError:
            loadorcomped = 'computed'
            if verbose:
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

            else:
                if inikryupd and tcur == t0:
                    print('\n1st step direct solve to init krylov\n')
                    vp_new = spsla.spsolve(IterA, Iterrhs)
                    vp_old = np.atleast_2d(vp_new).T
                    # TolCor = 0
                    inikryupd = False  # only once !!
                else:
                    curls = krypy.linsys.LinearSystem(IterA, Iterrhs,
                                                      M=MInv)

                    # tstart = time.time()

                    # extrapolating the initial value
                    upv = (vp_old - vp_oldold)

                    ret = krypy.linsys.\
                        Minres(curls, x0=vp_old + 0*upv, tol=linatol,
                               store_arnoldi=False, maxiter=1500)
                    # RestartedGmres(curls, x0=vp_old + upv,
                    #                tol=linatol,
                    #                maxiter=iniiterfac*MaxIter,
                    #                max_restarts=100)

                    # ret = krypy.linsys.\
                    #     Minres(curls, maxiter=20*TsP.MaxIter,
                    #            x0=vp_old + upv, tol=TolCor*TsP.linatol)
                    # tend = time.time()
                    vp_oldold = vp_old
                    vp_old = ret.xk

                    # print(('Needed {0} of max {4}*{1} iterations: ' +
                    #        'final relres = {2}\n TolCor was {3}').
                    #       format(len(ret.resnorms), MaxIter,
                    #              ret.resnorms[-1], 1, iniiterfac))
                    # print('Elapsed time {0}'.format(tend - tstart))
                    # iniiterfac = 1  # fac only in the first Krylov Call

            np.save(cdatstr + '_v', vp_old[:Nv])
            np.save(cdatstr + '_p', PFacI*vp_old[Nv:])

        dictofvstrs.update({tcur: cdatstr + '_v'})
        dictofpstrs.update({tcur: cdatstr + '_p'})

        if np.mod(tk+1, np.int(np.floor(Nts/numoutputpts))) == 0:
            plotroutine(vp_old, t=tcur)
            print('{0}/{1} time steps {2}'.format(tk, Nts, loadorcomped))

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


def _setuptdisc(t0=None, tE=None, trange=None, Nts=None):

    try:
        t0, tE, Nts = trange[0], trange[-1], trange.size-1
    except TypeError:
        trange = np.linspace(t0, tE, Nts+1)

    dt = (tE - t0)/Nts
    dtvec = trange[1:] - trange[:-1]
    if not np.allclose(np.linalg.norm(dtvec)/np.sqrt(dtvec.size), dt):
        raise UserWarning('trange not equispaced -- cannot prefac A')

    return t0, tE, Nts, dt
