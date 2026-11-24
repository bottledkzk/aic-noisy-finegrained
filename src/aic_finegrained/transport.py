"""FP64 semi-relaxed KL transport: hard sample rows, soft class masses."""
import time
import numpy as np
from scipy.special import logsumexp


def solve(scores, rho=9.0, temperature=1.0, tol=1e-10, max_iter=2000, progress=None):
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or min(s.shape) < 1 or not np.isfinite(s).all():
        raise ValueError('A finite nonempty N by C score matrix is required')
    if not np.isfinite(rho) or rho < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('rho >= 0 and temperature > 0 must be finite')
    if not 0 < tol < 1 or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('Positive iteration budget and tolerance in (0,1) required')
    start = time.perf_counter()
    k = s / temperature
    k = k - k.max(axis=1, keepdims=True)
    n, c = k.shape
    tau = rho / (1.0 + rho)
    v = np.zeros(c, dtype=np.float64)
    target_log = np.log(n/c)
    converged = False
    for step in range(1, max_iter + 1):
        logu = -logsumexp(k + v[None, :], axis=1)
        logbase = logsumexp(k + logu[:, None], axis=0)
        proposed = tau * (target_log - logbase)
        # A common shift of v does not change any row probability.
        proposed -= proposed.mean()
        change = float(np.max(np.abs(proposed - v)))
        v = proposed
        if progress and (step == 1 or step % 25 == 0):
            progress({'iteration':step,'centered_scale_change':change,'seconds':time.perf_counter()-start})
        if change <= tol:
            qlog = k + v[None, :]
            qlog -= logsumexp(qlog, axis=1, keepdims=True)
            q = np.exp(qlog)
            mass = q.sum(axis=0, dtype=np.float64)/n
            residual = v + rho*np.log(mass*c)
            kkt = float(np.ptp(residual))
            if kkt <= max(1e-9, 100*tol):
                converged = True
                break
    if not converged:
        raise RuntimeError(f'Semi-relaxed OT did not converge: iteration={step}, scale_change={change}')
    row_error = float(np.max(np.abs(q.sum(axis=1)-1)))
    if not np.isfinite(q).all() or row_error > 1e-12:
        raise RuntimeError('Nonfinite probability or row residual exceeds 1e-12')
    info = {'status':'PASS','rho':rho,'temperature':temperature,'tau':tau,'iterations':step,
            'scale_change':change,'KKT_class_residual_range':kkt,'row_abs_error':row_error,
            'soft_class_relative_deviation_from_uniform':float(np.max(np.abs(mass*c-1))),
            'soft_class_mass_min':float(mass.min()*n),'soft_class_mass_max':float(mass.max()*n),
            'seconds':time.perf_counter()-start,'dtype':'float64','device':'cpu',
            'n_rows':n,'n_classes':c,'marginal_constraint':'hard row sum1, KL-soft column probability mass'}
    return q, v, info


def self_test():
    from scipy.optimize import minimize
    rng = np.random.default_rng(431)
    s = rng.normal(size=(9,4))*1.3
    logp = s-logsumexp(s,axis=1,keepdims=True)
    p = np.exp(logp)
    q, v, info = solve(s)
    n,c=s.shape; rho=9.0
    def fun(x):
        a=x.reshape(n,c);m=a.mean(axis=0)
        f=(a*(np.log(a)-logp)).sum()/n+rho*(m*np.log(m*c)).sum()
        g=(np.log(a)-logp+1+rho*(np.log(m*c)+1)[None,:])/n
        return f,g.ravel()
    eq = np.kron(np.eye(n),np.ones((1,c)))
    result = minimize(fun,p.ravel(),jac=True,method='SLSQP',bounds=[(1e-14,1)]*(n*c),
      constraints={'type':'eq','fun':lambda x:eq@x-1,'jac':lambda x:eq},
      options={'ftol':1e-12,'maxiter':2000})
    assert result.success, result.message
    err=float(np.max(np.abs(result.x.reshape(n,c)-q)))
    assert err<2e-6,err
    zero,_,_=solve(s,rho=0)
    assert np.max(np.abs(zero-p))<1e-13
    row_order=rng.permutation(n);col_order=rng.permutation(c)
    perm,_,_=solve(s[row_order][:,col_order])
    assert np.max(np.abs(perm-q[row_order][:,col_order]))<1e-11
    shifted,_,_=solve(s+rng.normal(size=(n,1))*1e3)
    assert np.max(np.abs(shifted-q))<1e-11
    # Independent double-precision multiplicative hard-balanced endpoint.
    kernel=np.exp(s);vv=np.ones(c)
    for _ in range(5000):
        uu=1/(kernel@vv);vv=(n/c)/(kernel.T@uu)
    hard=uu[:,None]*kernel*vv[None,:]
    large,_,_=solve(s,rho=1e6,tol=1e-12)
    limiterr=float(np.max(np.abs(large-hard)))
    assert limiterr<2e-6,limiterr
    for bad in [np.empty((0,4)),np.zeros(4),np.array([[np.nan]])]:
        try:solve(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid scores accepted')
    for bad in [-1,np.inf,np.nan]:
        try:solve(s,rho=bad)
        except ValueError:pass
        else:raise AssertionError('Invalid rho accepted')
    try:solve(s,max_iter=1,tol=1e-12)
    except RuntimeError:pass
    else:raise AssertionError('Nonconvergence accepted')
    return {'status':'PASS','independent_primal_SLSQP_max_error':err,
      'large_rho_vs_independent_balanced_max_error':limiterr,
      'KKT_residual_range':info['KKT_class_residual_range'],
      'rho0_permutations_row_offsets_invalid_input_nonconvergence':'PASS',
      'scope':'synthetic mathematical engineering only; no quality evidence'}

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
