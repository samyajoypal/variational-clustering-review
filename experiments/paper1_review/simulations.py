"""Fresh positive-noise S2/S3 simulations. No legacy result is read.

Run an approved workload entry with --index and a new --output directory.
Finite quadrature is checked numerically, never called a certified ELBO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, roots_genlaguerre
from scipy.stats import norm, t

from experiments.paper1_review.provenance import ROOT, finish_run, start_run, write_json
from experiments.paper1_review.workload import plan

LOG2PI = math.log(2 * math.pi)
SUPPORTED = {"gaussian_q_student_prior", "structured_q_student_prior",
             "independent_gaussian_gamma", "coupled_gaussian_gamma", "gaussian",
             "student_q_student_prior", "direct_student", "full_covariance_gaussian"}
# These settings are frozen before final job launch and written into every run.
NUMERICS = dict(nu_bounds=[2.1, 200.], posterior_nu_bounds=[2.1, 200.],
                numerical_revision="20261028-scaled-reference-laplace",
                optimization_orders=[24, 64], refinement_orders=[48, 128],
                validation_orders=[64, 192], evidence_orders=[256, 512],
                standardized_log_tau_bounds=[-30., 10.],
                conditional_expectation_quadrature="Gaussian quadratic-form Laplace transform",
                conditional_laplace_orders=[96,192,256],
                conditional_log_laplace_bounds=[-36.,8.],
                evidence_log_tau_bounds="union of prior and zero-noise conditional Gamma log-mean minus30sd/plus10sd",
                full_covariance_log_laplace_bounds=[-30., 8.],
                max_iterations=250, refinement_iterations=150, ftol=1e-11, gtol=2e-7,
                maximum_quadrature_error=.0001, mean_quadrature_error=.00001,
                reference_error=2e-7, elbo_excess_tolerance=.0002,
                optimizer_projected_gradient_tolerance=.002,
                starts_prior_nu=[5., 30.], starts_posterior_nu=[8., 40.],
                objective_selection="highest refined training objective; explicit Gaussian-limit candidate for S2",
                quadrature_claim="attained-parameter numerical refinement, not a rigorous integration certificate")


def covariance_scale(nu):
    """Student scale with unit covariance; differentiable for torch inputs."""
    return (nu - 2.) / nu


def generate_data(config, *, smoke=False):
    """Seed keys omit m/family, pairing observations across nuisance dimensions."""
    n = config.get("smoke_size",16) if smoke else config["n_fit"]
    nh = 32 if smoke else config.get("n_heldout", 4096)
    result = {}
    for split, size, key in [("fit", n, 17), ("heldout", nh, 29)]:
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], key]))
        truth = config["truth"]
        if config["module"] == "S3":
            label = rng.integers(0, 2, size)
            dfs = {"t3_t30": [3., 30.], "t5_t5": [5., 5.],
                   "gaussian_gaussian": [np.inf, np.inf]}[truth]
            nu = np.asarray(dfs)[label]
            signal = rng.standard_normal(size)
            finite = np.isfinite(nu)
            signal[finite] *= np.sqrt(covariance_scale(nu[finite]) /
                                       rng.gamma(nu[finite] / 2, 2 / nu[finite]))
            signal += np.asarray([-1., 1.])[label]
            result[f"{split}_labels"] = label
        else:
            signal = rng.standard_normal(size)
            if truth == "t5":
                signal *= np.sqrt(.6 / rng.gamma(2.5, .4, size))
        result[f"{split}_signal"] = signal
        result[f"{split}_x"] = signal + config["noise_sd"] * rng.standard_normal(size)
    return result


@lru_cache(maxsize=32)
def _legendre(order):
    v, w = leggauss(order)
    lo, hi = NUMERICS["standardized_log_tau_bounds"]
    return (v + 1) * (hi - lo) / 2 + lo, w * (hi - lo) / 2


def gamma_grid(shape, rate, order):
    """Differentiable standardized-log-Gamma quadrature, including Jacobian."""
    vv, ww = _legendre(order)
    v = torch.as_tensor(vv, dtype=shape.dtype, device=shape.device)
    w = torch.as_tensor(ww, dtype=shape.dtype, device=shape.device)
    location = torch.digamma(shape) - torch.log(rate)
    spread = torch.sqrt(torch.polygamma(1, shape))
    u = location[..., None] + spread[..., None] * v
    tau = u.exp()
    logw = (shape[..., None] * torch.log(rate)[..., None] - torch.lgamma(shape)[..., None]
            + shape[..., None] * u - rate[..., None] * tau
            + torch.log(spread)[..., None] + w.log())
    return tau, logw


def evidence_torch(x, nu, noise_sd, *, mu=0., order=256):
    """Scalar observed density; invariant to ignored latent coordinates.

    Include the prior and zero-noise conditional precision ranges. A grid
    centered only on the prior can miss tiny likelihood-weighted mass for an
    extreme x even when successive orders agree. These are still numerical
    truncation bounds, independently checked in signal space, not a certificate.
    """
    a=nu/2;scale=covariance_scale(nu)
    prior_location=torch.digamma(a)-a.log()
    prior_sd=torch.sqrt(torch.polygamma(1,a))
    posterior_a=a+.5
    posterior_b=a+(x-mu).square()/(2*scale)
    posterior_location=torch.digamma(posterior_a)-posterior_b.log()
    posterior_sd=torch.sqrt(torch.polygamma(1,posterior_a))
    lo=torch.minimum(prior_location-30*prior_sd,posterior_location-30*posterior_sd)
    hi=torch.maximum(prior_location+10*prior_sd,posterior_location+10*posterior_sd)
    nodes,weights=leggauss(order)
    nodes=torch.as_tensor(nodes,dtype=x.dtype,device=x.device)
    weights=torch.as_tensor(weights,dtype=x.dtype,device=x.device)
    u=lo[...,None]+(nodes+1)*(hi-lo)[...,None]/2
    tau=u.exp()
    logw=a*a.log()-torch.lgamma(a)+a*u-a*tau+weights.log()+((hi-lo)/2).log()[...,None]
    variance = noise_sd ** 2 + scale / tau
    return torch.logsumexp(logw - .5 * (LOG2PI + variance.log()
                              + (x[..., None] - mu).square() / variance), -1)


def evidence(x, nu, noise_sd, *, mu=0., order=256):
    x = np.asarray(x, dtype=float)
    if np.isinf(nu):
        return norm.logpdf(x, loc=mu, scale=np.sqrt(1 + noise_sd ** 2))
    with torch.no_grad():
        return evidence_torch(torch.tensor(x, dtype=torch.float64),
                              torch.tensor(nu, dtype=torch.float64), noise_sd,
                              mu=mu, order=order).numpy()


def independent_evidence(x, nu, noise_sd, mu=0.):
    """Independent, relatively scaled direct signal integration.

    Integrating an unscaled tail density with an absolute tolerance can accept a
    poor *relative* error. Scale at the largest stationary log integrand instead.
    The stationary points solve a cubic, so a displaced narrow mode is not lost
    by an infinite-interval adaptive integral. This uses neither the Gamma scale
    mixture nor its quadrature nodes.
    """
    if np.isinf(nu):
        return float(norm.logpdf(x, loc=mu, scale=np.sqrt(1 + noise_sd ** 2)))
    scale = np.sqrt(covariance_scale(nu))
    delta = x-mu
    roots = np.roots([noise_sd**2, 2*delta*noise_sd,
                      delta**2+nu*scale**2+(nu+1)*noise_sd**2,
                      (nu+1)*noise_sd*delta])
    stationary = sorted(float(r.real) for r in roots if abs(r.imag) <= 1e-9*(1+abs(r.real)))
    def log_integrand(e):
        return t.logpdf(x+noise_sd*e,nu,loc=mu,scale=scale)+norm.logpdf(e)
    shift = max(float(log_integrand(e)) for e in [0.,*stationary])
    endpoints = [-np.inf,*stationary,np.inf]
    pieces = [quad(lambda e: np.exp(log_integrand(e)-shift),left,right,
                   epsabs=1e-13,epsrel=1e-11,limit=300)
              for left,right in zip(endpoints[:-1],endpoints[1:])]
    value,error = map(sum,zip(*pieces))
    if not np.isfinite(value) or value <= 0 or error > 1e-8*value:
        raise ArithmeticError(f"Scaled direct integration failed relative-error guard: {value=}, {error=}")
    return float(shift+np.log(value))


def check_reference(x, nu, sd, mu=0.):
    coarse = evidence(x, nu, sd, mu=mu, order=256)
    fine = evidence(x, nu, sd, mu=mu, order=512)
    selected = np.unique(np.r_[np.linspace(0, len(x)-1, min(9, len(x)), dtype=int),
                               np.argmin(x), np.argmax(x)])
    independent = [abs(independent_evidence(float(x[i]), nu, sd, mu)-fine[i]) for i in selected]
    record = dict(refinement_max_error=float(np.max(np.abs(coarse-fine))),
                  direct_signal_max_error=float(max(independent)), independent_points=len(selected))
    if max(record["refinement_max_error"], record["direct_signal_max_error"]) > NUMERICS["reference_error"]:
        raise ArithmeticError(f"Scalar evidence accuracy gate failed: {record}")
    return fine, record


class ConditionalObjective:
    """Observation-specific conditional inference with deterministic quadrature.

    Gaussian families have a free signal mean/variance and isotropic nuisance
    variance, zero nuisance mean. Direct Student additionally has a free df per
    observation; its variance parameters are t scale parameters, not covariances.
    Independent Gamma is analytically coordinate-optimized, never sampled.
    Coupled Gamma has free shape/rate; ignored conditional scales equal the prior,
    which is the exact minimizer within that coupled family.
    """
    def __init__(self, x, m, sd, family, *, fixed_nu=None, mu=0., order=24,
                 tau_order=64, device="cpu", chunk=64, quadrature="radial",laplace_order=None):
        self.x = torch.as_tensor(x, dtype=torch.float64, device=device)
        self.n, self.m, self.d = len(x), m, m + 1
        self.sd, self.family, self.fixed_nu, self.mu = sd, family, fixed_nu, mu
        self.order, self.tau_order, self.chunk = order, tau_order, chunk
        if quadrature not in {"radial","laplace"}:
            raise ValueError("quadrature must be radial or laplace")
        self.quadrature=quadrature
        if quadrature=="laplace":
            v,w=leggauss(laplace_order or tau_order)
            lo,hi=NUMERICS["conditional_log_laplace_bounds"]
            self.laplace_t=torch.as_tensor(np.exp(lo+(v+1)*(hi-lo)/2),dtype=self.x.dtype,device=device)
            self.laplace_w=torch.as_tensor(w*(hi-lo)/2,dtype=self.x.dtype,device=device)
        h, hw = hermgauss(order)
        self.h = torch.as_tensor(h*np.sqrt(2), dtype=self.x.dtype, device=device)
        self.hw = torch.as_tensor(hw/np.sqrt(np.pi), dtype=self.x.dtype, device=device)
        if m:
            r, rw = roots_genlaguerre(order, m/2-1)
            r, rw = 2*r, rw*np.exp(-gammaln(m/2))
        else:
            r, rw = np.zeros(1), np.ones(1)
        self.r = torch.as_tensor(r, dtype=self.x.dtype, device=device)
        self.rw = torch.as_tensor(rw, dtype=self.x.dtype, device=device)

    @property
    def direct(self):
        return self.family in {"student_q_student_prior", "direct_student"}

    @property
    def coupled(self):
        return self.family == "coupled_gaussian_gamma"

    def initial(self, start):
        n, sd = self.n, self.sd
        p = [] if self.fixed_nu is not None else [np.log(NUMERICS["starts_prior_nu"][start]-2)]
        bounds = [] if self.fixed_nu is not None else [(np.log(.1), np.log(198.))]
        x = self.x.cpu().numpy()
        mean = (x + sd**2*self.mu)/(1+sd**2)
        # Distinct predetermined starts even when the prior df is fixed (S3).
        if start:
            mean = self.mu + .75*(mean-self.mu)
        signal_variance = sd**2/(1+sd**2)*(2. if start else 1.)
        p.extend(mean.tolist()); bounds += [(None, None)]*n
        p.extend([np.log(signal_variance)]*n); bounds += [(-12., 8.)]*n
        if self.m and not self.coupled:
            p.extend([math.log(2.) if start else 0.]*n); bounds += [(-12., 8.)]*n
        if self.direct:
            p.extend([np.log(NUMERICS["starts_posterior_nu"][start]-2)]*n)
            bounds += [(np.log(.1), np.log(198.))]*n
        if self.coupled:
            # q tau shape >1 ensures finite signal reconstruction variance.
            a = NUMERICS["starts_posterior_nu"][start]/2
            p.extend([np.log(a-1)]*n); bounds += [(np.log(.05), np.log(499.))]*n
            p.extend([np.log(a)]*n); bounds += [(np.log(.01), np.log(500.))]*n
        return np.asarray(p), bounds

    def unpack(self, p, sl):
        offset = int(self.fixed_nu is None)
        nu = 2+p[0].exp() if offset else p.new_tensor(self.fixed_nu)
        blocks = p[offset:].reshape(-1, self.n)
        mean, lv = blocks[0, sl], blocks[1, sl]
        pos = 2
        if self.m and not self.coupled:
            ln = blocks[pos, sl]; pos += 1
        else:
            ln = torch.zeros_like(mean)
        extra = blocks[pos:, sl]
        return nu, mean, lv, ln, extra

    def values(self, p, sl):
        nu, mean, lv, ln, extra = self.unpack(p, sl)
        x, m, d = self.x[sl], self.m, self.d
        vs, vn = lv.exp(), ln.exp()
        gaussian_prior = bool(self.fixed_nu is not None and np.isinf(self.fixed_nu))
        scale = p.new_tensor(1.) if gaussian_prior else covariance_scale(nu)
        entropy = .5*(d*(1+LOG2PI)+lv+m*ln)
        if self.coupled:
            aq, bq = 1+extra[0].exp(), extra[1].exp()
            a = nu/2
            klgamma = ((aq-a)*torch.digamma(aq)-torch.lgamma(aq)+torch.lgamma(a)
                       +a*(bq.log()-a.log())+aq*(a/bq-1))
            # Nuisance conditional covariance exactly scale/tau, KL contribution 0.
            klcond = .5*(vs/scale-1+scale.log()-lv+(aq/bq)*(mean-self.mu).square()/scale)
            rec = -.5*(LOG2PI+math.log(self.sd**2)+((x-mean).square()+vs*bq/(aq-1))/self.sd**2)
            return rec-klgamma-klcond
        if self.direct:
            qnu = 2+extra[0].exp()
            variance = vs*qnu/(qnu-2)
            rec = -.5*(LOG2PI+math.log(self.sd**2)+((x-mean).square()+variance)/self.sd**2)
            entropy = (torch.lgamma(qnu/2)-torch.lgamma((qnu+d)/2)
                       +.5*(d*torch.log(qnu*math.pi)+lv+m*ln)
                       +(qnu+d)/2*(torch.digamma((qnu+d)/2)-torch.digamma(qnu/2)))
            if gaussian_prior:
                expected_radius = (mean-self.mu).square()+(vs+m*vn)*qnu/(qnu-2)
                return rec+entropy-.5*(d*LOG2PI+expected_radius)
            tau, logw = gamma_grid(qnu/2, qnu/2, self.tau_order)
            # Unnormalized quadrature: no hidden renormalization changes objective.
            if self.quadrature=="laplace":
                elog=self.laplace_elog((mean-self.mu)[:,None],vs[:,None]/tau,vn[:,None]/tau,nu)
            else:
                z = mean[:, None, None]+torch.sqrt(vs[:, None, None]/tau[..., None])*self.h
                radius = ((z-self.mu).square()[..., None]
                          +vn[:, None, None, None]/tau[:, :, None, None]*self.r)
                elog = (torch.log1p(radius/(nu*scale))*self.hw[None,None,:,None]
                        *self.rw[None,None,None,:]).sum((-1,-2))
            elog = (elog*logw.exp()).sum(-1)
        else:
            rec = -.5*(LOG2PI+math.log(self.sd**2)+((x-mean).square()+vs)/self.sd**2)
            if gaussian_prior:
                return rec+entropy-.5*(d*LOG2PI+(mean-self.mu).square()+vs+m*vn)
            if self.family == "independent_gaussian_gamma":
                elog = torch.log1p(((mean-self.mu).square()+vs+m*vn)/(nu*scale))
            elif self.quadrature=="laplace":
                elog=self.laplace_elog(mean-self.mu,vs,vn,nu)
            else:
                z = mean[:, None]+torch.sqrt(vs[:, None])*self.h
                radius = (z-self.mu).square()[...,None]+vn[:,None,None]*self.r
                elog = (torch.log1p(radius/(nu*scale))*self.hw[None,:,None]*self.rw).sum((-1,-2))
        constant = (torch.lgamma((nu+d)/2)-torch.lgamma(nu/2)
                    -.5*d*torch.log(nu*math.pi*scale))
        return rec+entropy+constant-(nu+d)/2*elog

    def laplace_elog(self,delta,vs,vn,nu):
        """E log(1+Q/(nu-2)) for a conditional diagonal Gaussian.

        Tonelli and log(1+y)=integral exp(-t)*(1-exp(-ty))*dt/t
        yield the same continuous expectation as radial Hermite/Laguerre.
        Integrating over log(t) avoids its removable endpoint singularity.
        """
        u=self.laplace_t/(nu*covariance_scale(nu))
        denominator=1+2*u*vs[...,None]
        logmgf=(-.5*torch.log1p(2*u*vs[...,None])
                -.5*self.m*torch.log1p(2*u*vn[...,None])
                -u*delta[...,None].square()/denominator)
        return (-torch.expm1(logmgf)*torch.exp(-self.laplace_t)*self.laplace_w).sum(-1)

    def laplace_lower_tail_bound(self,parameters):
        """Attained-parameter bound on omitted lower-log-t ELBO magnitude.

        1-L_Q(t/a)<=t E[Q]/a gives exp(lo)*E[Q]/a. The upper
        tail above exp(8) is smaller than exp(-exp(8))/exp(8).
        This bounds truncation only, not finite-node quadrature error.
        """
        p=torch.as_tensor(parameters,dtype=self.x.dtype,device=self.x.device)
        nu,mean,lv,ln,extra=self.unpack(p,slice(None))
        radial=lv.exp()+self.m*ln.exp()
        if self.direct:
            qnu=2+extra[0].exp()
            radial=radial*qnu/(qnu-2)
        radius=(mean-self.mu).square()+radial
        bound=(nu+self.d)/2*math.exp(NUMERICS["conditional_log_laplace_bounds"][0])*radius/(nu*covariance_scale(nu))
        return float(bound.max().detach().cpu())

    def __call__(self, parameters, grad=True, individual=False,return_values_and_gradient=False):
        if return_values_and_gradient and not grad:
            raise ValueError("Returning gradients requires grad=True")
        p = torch.tensor(parameters, dtype=torch.float64, device=self.x.device, requires_grad=grad)
        values = []
        for begin in range(0, self.n, self.chunk):
            v = self.values(p, slice(begin, min(begin+self.chunk, self.n)))
            if grad:
                (-v.sum()/self.n).backward()
            values.append(v.detach().cpu().numpy())
        values = np.concatenate(values)
        if return_values_and_gradient:
            return values,p.grad.cpu().numpy()
        if individual:
            return values
        if grad:
            return -float(values.mean()), p.grad.cpu().numpy()
        return -float(values.mean())


class FullCovarianceObjective(ConditionalObjective):
    """Free mean and Cholesky covariance, with scalar Laplace quadrature.

    For Q=||Z-mu||² and Z Gaussian, E log(1+Q/a) equals
    integral exp(-t)*(1-E exp(-t Q/a))*dt/t (Tonelli). The Gaussian
    quadratic-form Laplace transform is available for arbitrary covariance.
    This does not assume that a globally best covariance is diagonal.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tri = torch.tril_indices(self.d,self.d,device=self.x.device)
        self.diagonal = self.tri[0] == self.tri[1]
        v,w=leggauss(self.tau_order)
        lo,hi=NUMERICS["full_covariance_log_laplace_bounds"]
        self.laplace_t=torch.as_tensor(np.exp(lo+(v+1)*(hi-lo)/2),dtype=self.x.dtype,device=self.x.device)
        self.laplace_w=torch.as_tensor(w*(hi-lo)/2,dtype=self.x.dtype,device=self.x.device)

    def initial(self,start):
        n,d,sd=self.n,self.d,self.sd
        x=self.x.cpu().numpy()
        means=np.zeros((n,d));means[:,0]=x/(1+sd**2)*(1. if not start else .75)
        scales=np.eye(d);scales[0,0]=np.sqrt(sd**2/(1+sd**2)*(1. if not start else 2.))
        if start:
            scales[1:,0]=.05
        row,col=self.tri.cpu().numpy()
        chol=scales[row,col];diagonal=row==col
        chol[diagonal]=np.log(chol[diagonal])
        blocks=np.concatenate([means,np.tile(chol,(n,1))],axis=1)
        p=np.r_[np.log(NUMERICS["starts_prior_nu"][start]-2),blocks.ravel()]
        blockbounds=[(None,None)]*d+[(-6.,4.) if isdiag else (-20.,20.) for isdiag in diagonal]
        return p,[(np.log(.1),np.log(198.))]+blockbounds*n

    def values(self,p,sl):
        nu=2+p[0].exp();scale=covariance_scale(nu)
        block=p[1:].reshape(self.n,-1)[sl]
        mean=block[:,:self.d]
        raw=block[:,self.d:]
        entries=torch.where(self.diagonal[None,:],raw.exp(),raw)
        chol=torch.zeros((len(block),self.d,self.d),dtype=p.dtype,device=p.device)
        chol[:,self.tri[0],self.tri[1]]=entries
        covariance=chol@chol.transpose(-1,-2)
        rec=-.5*(LOG2PI+math.log(self.sd**2)+((self.x[sl]-mean[:,0]).square()+covariance[:,0,0])/self.sd**2)
        entropy=.5*self.d*(1+LOG2PI)+raw[:,self.diagonal].sum(-1)
        u=self.laplace_t/(nu*scale)
        eye=torch.eye(self.d,dtype=p.dtype,device=p.device)
        matrix=eye+2*u[None,:,None,None]*covariance[:,None,:,:]
        factor=torch.linalg.cholesky(matrix)
        centered=mean.clone();centered[:,0]=centered[:,0]-self.mu
        solved=torch.linalg.solve_triangular(factor,centered[:,None,:,None].expand(-1,len(u),-1,-1),upper=False)
        logmgf=-factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)-u*solved.square().sum((-1,-2))
        elog=(-torch.expm1(logmgf)*torch.exp(-self.laplace_t)*self.laplace_w).sum(-1)
        constant=torch.lgamma((nu+self.d)/2)-torch.lgamma(nu/2)-.5*self.d*torch.log(nu*math.pi*scale)
        return rec+entropy+constant-(nu+self.d)/2*elog


def projected_gradient(parameters, gradient, bounds):
    g = np.array(gradient, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and parameters[i] <= lo+1e-7 and g[i] > 0:
            g[i] = 0
        if hi is not None and parameters[i] >= hi-1e-7 and g[i] < 0:
            g[i] = 0
    return float(np.max(np.abs(g)))


def optimize_conditional(x, m, sd, family, *, fixed_nu=None, mu=0., device="cpu", smoke=False,save_starts=None):
    common = dict(x=x, m=m, sd=sd, family=family, fixed_nu=fixed_nu, mu=mu, device=device)
    objective_class=FullCovarianceObjective if family=="full_covariance_gaussian" else ConditionalObjective
    conditional_laplace=family in {"gaussian_q_student_prior","gaussian","student_q_student_prior","direct_student"}
    options_kernel=dict(quadrature="laplace") if conditional_laplace else {}
    coarse = objective_class(**common, order=24, tau_order=64,laplace_order=96,**options_kernel)
    fine = objective_class(**common, order=48, tau_order=128,laplace_order=192,**options_kernel)
    validation = objective_class(**common, order=64, tau_order=192,laplace_order=256,**options_kernel)
    fits = []
    if save_starts is not None:
        save_starts.mkdir(parents=True,exist_ok=False)
    for start in range(2):
        t0 = time.perf_counter()
        p0, bounds = coarse.initial(start)
        options = dict(maxiter=5 if smoke else NUMERICS["max_iterations"],
                       ftol=NUMERICS["ftol"], gtol=NUMERICS["gtol"], maxls=30, maxcor=15)
        first = minimize(coarse, p0, jac=True, method="L-BFGS-B", bounds=bounds, options=options)
        options["maxiter"] = 3 if smoke else NUMERICS["refinement_iterations"]
        fitted = minimize(fine, first.x, jac=True, method="L-BFGS-B", bounds=bounds, options=options)
        vals = fine(fitted.x, grad=False, individual=True)
        checked,validation_gradient = validation(fitted.x,return_values_and_gradient=True)
        error = np.abs(vals-checked)
        pg = projected_gradient(fitted.x, validation_gradient, bounds)
        # The objective is a mean; undo 1/N only for observation-specific params.
        # Otherwise increasing N would spuriously relax stationarity validation.
        scaled_gradient = validation_gradient.copy()
        scaled_gradient[int(fixed_nu is None):] *= len(x)
        per_observation_pg = projected_gradient(fitted.x, scaled_gradient, bounds)
        record = dict(start=start, prior_start_nu=NUMERICS["starts_prior_nu"][start],
                      mean_objective=float(checked.mean()), maximum_quadrature_error=float(error.max()),
                      mean_quadrature_error=float(error.mean()), optimizer_success=bool(fitted.success),
                      optimizer_message=str(fitted.message), projected_gradient=pg,
                      observation_rescaled_projected_gradient=per_observation_pg,
                      iterations=int(first.nit+fitted.nit), function_evaluations=int(first.nfev+fitted.nfev),
                      seconds=time.perf_counter()-t0)
        record["conditional_expectation_quadrature"]="laplace" if conditional_laplace else "unchanged_family_specific"
        record["laplace_lower_tail_bound"]=validation.laplace_lower_tail_bound(fitted.x) if conditional_laplace else None
        record["accuracy_passed"] = (record["maximum_quadrature_error"] <= NUMERICS["maximum_quadrature_error"]
                                     and record["mean_quadrature_error"] <= NUMERICS["mean_quadrature_error"])
        if conditional_laplace:
            record["accuracy_passed"] &= record["laplace_lower_tail_bound"] <= NUMERICS["mean_quadrature_error"]
        record["stationarity_passed"] = per_observation_pg <= NUMERICS["optimizer_projected_gradient_tolerance"]
        if save_starts is not None:
            artifact=save_starts/f"start{start}.npz"
            np.savez_compressed(artifact,initial_parameters=p0,coarse_parameters=first.x,
                                parameters=fitted.x,fine_objective=vals,validated_objective=checked,
                                fine_gradient=fitted.jac,validation_gradient=validation_gradient)
            record["start_artifact"]=f"{save_starts.name}/{artifact.name}"
        fits.append((record, fitted.x, checked))
    best = max(range(2), key=lambda j: fits[j][0]["mean_objective"])
    return fits[best][1], fits[best][2], dict(selected_start=best, starts=[f[0] for f in fits],
                                             accuracy_passed=all(f[0]["accuracy_passed"] for f in fits),
                                             stationarity_passed=all(f[0]["stationarity_passed"] for f in fits))


def fit_structured(x, sd, *, smoke=False, device="cpu",save_starts=None):
    """Exact site q=p(z,tau|x); scalar evidence is the algebraically same ELBO.

    Site precision is 1/sd² in the signal coordinate and exactly zero elsewhere.
    Thus no variational optimizer or nuisance-dimensional numerical integral is
    required. This is an exact-model reference, not a neural encoder evaluation.
    """
    xt = torch.tensor(x, dtype=torch.float64, device=device)
    def fun(p):
        raw = torch.tensor(p, dtype=torch.float64, device=device, requires_grad=True)
        nu = 2+raw[0].exp()
        loss = -evidence_torch(xt, nu, sd, order=256).mean()
        loss.backward()
        return float(loss.detach()), raw.grad.cpu().numpy()
    fits = []
    if save_starts is not None:
        save_starts.mkdir(parents=True,exist_ok=False)
    for start,nu0 in enumerate(NUMERICS["starts_prior_nu"]):
        fit = minimize(fun, [np.log(nu0-2)], jac=True, method="L-BFGS-B",
                       bounds=[(np.log(.1), np.log(198.))],
                       options=dict(maxiter=5 if smoke else 150, ftol=1e-13, gtol=1e-9))
        nu = float(2+np.exp(fit.x[0]))
        vals, checks = check_reference(x, nu, sd)
        record=dict(initial_nu=nu0, nu=nu, objective=float(vals.mean()),
                                  success=bool(fit.success), iterations=int(fit.nit),
                                  message=str(fit.message), **checks)
        if save_starts is not None:
            artifact=save_starts/f"start{start}.npz"
            np.savez_compressed(artifact,initial_parameters=np.array([np.log(nu0-2)]),parameters=fit.x,
                                nu=np.array(nu),validated_objective=vals,gradient=fit.jac)
            record["start_artifact"]=f"{save_starts.name}/{artifact.name}"
        fits.append((nu,vals,record))
    best = max(range(2), key=lambda i: fits[i][2]["objective"])
    return fits[best][0], fits[best][1], dict(selected_start=best, starts=[f[2] for f in fits],
                                             accuracy_passed=True,
                                             stationarity_passed=all(f[2]["success"] for f in fits))


def run_s2(config, data, output, *, smoke, device):
    x, test = data["fit_x"], data["heldout_x"]
    m, sd, family = config["nuisance_dimension"], config["noise_sd"], config["family"]
    if family == "structured_q_student_prior":
        nu, values, optimization = fit_structured(x, sd, smoke=smoke, device=device,save_starts=output/"optimizer_starts")
        parameters = np.asarray([nu])
    else:
        parameters, values, optimization = optimize_conditional(x, m, sd, family, device=device, smoke=smoke,
                                                               save_starts=output/"optimizer_starts")
        nu = float(2+np.exp(parameters[0]))
    finite_nu = nu
    finite_evidence, refcheck = check_reference(x, nu, sd)
    gaussian_values = evidence(x, np.inf, sd)
    selected_gaussian = bool(gaussian_values.mean() > values.mean())
    if selected_gaussian:
        nu, selected_values = np.inf, gaussian_values
    else:
        selected_values = values
    train_evidence, selectedcheck = check_reference(x, nu, sd)
    test_evidence, heldcheck = check_reference(test, nu, sd)
    max_excess = float(np.max(values-finite_evidence))
    accurate = optimization["accuracy_passed"] and max_excess <= NUMERICS["elbo_excess_tolerance"]
    result = dict(module=config["module"], family=family, nuisance_dimension=m,
                  selected_nu="gaussian_limit" if np.isinf(nu) else nu,
                  selected_gaussian_limit=selected_gaussian, finite_candidate_nu=finite_nu,
                  finite_candidate_mean_objective=float(values.mean()),
                  gaussian_limit_mean_objective=float(gaussian_values.mean()),
                  selected_mean_objective=float(selected_values.mean()),
                  selected_train_log_likelihood=float(train_evidence.mean()),
                  independent_heldout_log_likelihood=float(test_evidence.mean()),
                  finite_candidate_mean_gap=float((finite_evidence-values).mean()),
                  selected_mean_gap=float((train_evidence-selected_values).mean()),
                  maximum_elbo_excess=max_excess, optimization=optimization,
                  reference_checks=dict(finite_candidate=refcheck, selected=selectedcheck, heldout=heldcheck),
                  accuracy_passed=accurate, stationarity_passed=optimization["stationarity_passed"],
                  covariance_matching="scale=(nu-2)/nu differentiated inside all Student objectives",
                  reference_scope="scalar linear positive-noise model; structured posterior equals exact posterior")
    np.savez_compressed(output/"fitted.npz", parameters=parameters, finite_objective=values,
                        selected_objective=selected_values, fit_log_evidence=train_evidence,
                        heldout_log_evidence=test_evidence)
    return result


def allocation_metrics(labels, responsibility):
    from sklearn.metrics import adjusted_rand_score, log_loss
    pred = responsibility.argmax(1)
    return dict(accuracy=float(np.mean(pred==labels)), ari=float(adjusted_rand_score(labels,pred)),
                log_loss=float(log_loss(labels,responsibility,labels=[0,1])),
                component_recall=[float(np.mean(pred[labels==j]==j)) for j in range(2)],
                predicted_component0_fraction=float(np.mean(pred==0)))


def run_s3(config, data, output, *, smoke, device):
    x, sd, m = data["fit_x"], config["noise_sd"], config["nuisance_dimension"]
    dfs = {"t3_t30": [3.,30.], "t5_t5": [5.,5.], "gaussian_gaussian": [np.inf,np.inf]}[config["truth"]]
    columns, reference, details = [], [], []
    for k, (nu, mu) in enumerate(zip(dfs, [-1.,1.])):
        exact, checks = check_reference(x, nu, sd, mu)
        if np.isinf(nu):
            # Both families include the exact Gaussian posterior as a limit.
            parameters, vals = np.asarray([]), exact.copy()
            opt = dict(accuracy_passed=True, stationarity_passed=True,
                       analytic_gaussian_limit=True, starts_evaluated=2,
                       explanation="Both predetermined starts map to the analytic Gaussian posterior optimum")
        else:
            parameters, vals, opt = optimize_conditional(x, m, sd, config["family"],
                                                         fixed_nu=nu, mu=mu, device=device, smoke=smoke,
                                                         save_starts=output/f"component_{k}_starts")
        np.savez_compressed(output/f"component_{k}.npz", parameters=parameters, objective=vals, exact=exact)
        columns.append(vals); reference.append(exact)
        details.append(dict(component=k, reference_checks=checks, optimization=opt,
                            mean_gap=float(np.mean(exact-vals)),
                            maximum_elbo_excess=float(np.max(vals-exact))))
    values, exact = np.column_stack(columns), np.column_stack(reference)
    posterior = np.exp(values-logsumexp(values,axis=1)[:,None])
    bayes = np.exp(exact-logsumexp(exact,axis=1)[:,None])
    np.savez_compressed(output/"assignments.npz", variational=posterior, exact_structured=bayes)
    accurate = all(d["optimization"]["accuracy_passed"] and
                   d["maximum_elbo_excess"] <= NUMERICS["elbo_excess_tolerance"] for d in details)
    return dict(module="S3", family=config["family"], nuisance_dimension=m,
                fitted=allocation_metrics(data["fit_labels"],posterior),
                exact_structured=allocation_metrics(data["fit_labels"],bayes),
                assignment_disagreement=float(np.mean(posterior.argmax(1)!=bayes.argmax(1))),
                responsibility_max_difference=float(np.max(np.abs(posterior-bayes))),
                components=details, accuracy_passed=accurate,
                stationarity_passed=all(d["optimization"]["stationarity_passed"] for d in details),
                labels_used_for="fresh synthetic generation and evaluation only; never optimization or start selection",
                scope="known generative mixture, observation-specific conditional inference; not amortized clustering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=int)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--smoke-size",type=int,
                        help="Run short smoke iterations at this batch size (up to manifest n_fit)")
    parser.add_argument("--validation-size",type=int,
                        help="Full optimizer on a smaller fresh development seed; never a final result")
    parser.add_argument("--list-supported",action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest else plan()
    configurations = manifest["optimized_simulations"]
    if args.list_supported:
        print(json.dumps({"supported_indices":[i for i,c in enumerate(configurations) if c["family"] in SUPPORTED],
                          "unsupported_indices":[i for i,c in enumerate(configurations) if c["family"] not in SUPPORTED]}))
        return
    if args.index is None or args.output is None or not 0 <= args.index < len(configurations):
        parser.error("--index in manifest range and a new --output directory are required")
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU timing will be reported as GPU timing")
    config = dict(configurations[args.index], index=args.index, smoke=args.smoke, device=args.device,
                  protocol_version=manifest["protocol_version"], numerics=NUMERICS,
                  hardware={"cuda_available":torch.cuda.is_available(),
                            "gpu":torch.cuda.get_device_name(0) if args.device == "cuda" else None},
                  sampling="SeedSequence([seed,17]) fit, [seed,29] independent heldout; excludes family and m")
    development = args.smoke or args.validation_size is not None
    if args.smoke_size is not None:
        if not args.smoke or not 2 <= args.smoke_size <= config["n_fit"]:
            parser.error("--smoke-size requires --smoke and must be between 2 and manifest n_fit")
        config["smoke_size"]=args.smoke_size
    if args.validation_size is not None:
        if args.smoke or not 2 <= args.validation_size <= config["n_fit"]:
            parser.error("--validation-size must be between 2 and n_fit and cannot accompany --smoke")
        config.update(n_fit=args.validation_size, n_heldout=256, validation_size=args.validation_size)
    if development:
        config.update(manifest_seed=config["seed"], seed=config["seed"]+100000)
    sources = [Path(__file__), ROOT/"experiments/paper1_review/workload.py",
               ROOT/"tests/test_paper1_review_simulations.py"]
    meta = start_run(args.output,stage="validation" if development else "deterministic_simulation",config=config,sources=sources)
    if args.device=="cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        if config["family"] not in SUPPORTED:
            raise NotImplementedError(f"Unsupported inference family {config['family']}; no result fabricated")
        data = generate_data(config,smoke=args.smoke)
        np.savez_compressed(args.output/"fresh_inputs.npz",**data)
        hashes = {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in data.items()}
        result = (run_s3 if config["module"] == "S3" else run_s2)(config,data,args.output,smoke=args.smoke,device=args.device)
        result.update(seconds=time.perf_counter()-started, data_array_sha256=hashes,
                      actual_n_fit=len(data["fit_x"]), actual_n_heldout=len(data["heldout_x"]),
                      smoke_only=args.smoke, validation_only=development, reused_empirical_outputs=False,
                      peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=="cuda" else None,
                      peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if args.device=="cuda" else None)
        write_json(args.output/"result.json",result)
        passed = result["accuracy_passed"] and result["stationarity_passed"]
        finish_run(args.output,meta,status="completed" if passed or args.smoke else "failed",
                   publishable_numerical_result=bool(passed and not development), elapsed_seconds=result["seconds"],
                   quality_gate_passed=passed, smoke_only=args.smoke)
        print(json.dumps({"index":args.index,"seconds":result["seconds"],"quality_gate_passed":passed,"smoke":args.smoke}))
        if not passed and not args.smoke:
            raise SystemExit(2)
    except Exception as error:
        finish_run(args.output,meta,status="failed",error=f"{type(error).__name__}: {error}",
                   elapsed_seconds=time.perf_counter()-started)
        raise


if __name__ == "__main__":
    main()
