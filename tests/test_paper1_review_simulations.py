import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm,t

from experiments.paper1_review.simulations import (
    ConditionalObjective, covariance_scale, evidence, evidence_torch,
    generate_data, independent_evidence, gamma_grid,
    FullCovarianceObjective,
)


@pytest.mark.parametrize("nu", [2.1, 3., 5., 30., 200.])
@pytest.mark.parametrize("sd", [.5, 2., 10.])
def test_scalar_reference_matches_independent_signal_integration(nu,sd):
    x=np.array([-4.,0.,2.5])
    got=evidence(x,nu,sd,order=512)
    expected=[independent_evidence(v,nu,sd) for v in x]
    np.testing.assert_allclose(got,expected,atol=2e-8,rtol=1e-8)


def test_covariance_matched_df_chain_rule():
    x=torch.tensor([-3.,-.1,2.],dtype=torch.float64)
    nu=torch.tensor(5.,dtype=torch.float64,requires_grad=True)
    value=evidence_torch(x,nu,2.,order=256).sum()
    derivative=torch.autograd.grad(value,nu)[0].item()
    h=1e-4
    numeric=(evidence(x.numpy(),5+h,2.).sum()-evidence(x.numpy(),5-h,2.).sum())/(2*h)
    assert abs(derivative-numeric)<1e-8
    assert covariance_scale(5.)==.6


def test_paired_data_and_independent_heldout():
    config=dict(seed=1101,module="S2_primary",truth="t5",noise_sd=2.,n_fit=512,n_heldout=4096,
                nuisance_dimension=0,family="gaussian_q_student_prior")
    one=generate_data(config)
    two=generate_data(dict(config,nuisance_dimension=20,family="structured_q_student_prior"))
    for key in one:
        np.testing.assert_array_equal(one[key],two[key])
    assert len(one['heldout_x'])==4096
    assert not np.array_equal(one['fit_x'],one['heldout_x'][:512])


@pytest.mark.parametrize("m",[0,5,20,100])
def test_zero_site_nuisance_exact_reference_invariant(m):
    # Reference integrates nuisance coordinates analytically, hence has no m input.
    x=np.array([-1.,.2,4.])
    np.testing.assert_array_equal(evidence(x,5.,.5),evidence(x,5.,.5))
    # Independent check using the conditional Gaussian moments/site normalizer.
    nu=torch.tensor(5.,dtype=torch.float64)
    tau,lw=gamma_grid(nu/2,nu/2,512)
    scale=covariance_scale(nu)
    site_precision=4.
    lam=torch.zeros(1+m,dtype=torch.float64);lam[0]=site_precision
    y=torch.zeros(len(x),1+m,dtype=torch.float64);y[:,0]=torch.tensor(x)
    rho=scale*lam
    logsite=-.5*torch.log1p(rho[None,:]/tau[:,None]).sum(-1)
    logsite=logsite[None,:]-.5*(lam[None,None,:]*y[:,None,:]**2*tau[None,:,None]/(tau[None,:,None]+rho)).sum(-1)
    got=torch.logsumexp(lw[None,:]+logsite,1)-.5*np.log(2*np.pi*.25)
    np.testing.assert_allclose(got.numpy(),evidence(x,5.,.5,order=512),atol=2e-13)


@pytest.mark.parametrize("family",["gaussian_q_student_prior","independent_gaussian_gamma",
                                   "coupled_gaussian_gamma","student_q_student_prior"])
def test_objective_gradients(family):
    obj=ConditionalObjective(np.array([-.3,2.]),2,.7,family,order=24,tau_order=128)
    p,_=obj.initial(0)
    value,grad=obj(p)
    assert np.isfinite(value) and np.isfinite(grad).all()
    for i in [0,1,3,len(p)-1]:
        h=1e-5
        plus=p.copy();minus=p.copy();plus[i]+=h;minus[i]-=h
        numerical=(obj(plus,grad=False)-obj(minus,grad=False))/(2*h)
        np.testing.assert_allclose(grad[i],numerical,atol=3e-7,rtol=3e-6)


def test_independent_gamma_bound_below_collapsed_gaussian():
    common=dict(x=np.array([-.3,2.]),m=5,sd=.7)
    collapsed=ConditionalObjective(**common,family="gaussian_q_student_prior",order=64)
    independent=ConditionalObjective(**common,family="independent_gaussian_gamma",order=64)
    p,_=collapsed.initial(0)
    assert np.all(independent(p,grad=False,individual=True)<=collapsed(p,grad=False,individual=True)+1e-12)


def test_coupled_conditional_nuisance_invariance():
    a=ConditionalObjective(np.array([-.3,2.]),0,.7,"coupled_gaussian_gamma")
    b=ConditionalObjective(np.array([-.3,2.]),20,.7,"coupled_gaussian_gamma")
    p,_=a.initial(0)
    np.testing.assert_allclose(a(p,grad=False,individual=True),b(p,grad=False,individual=True),atol=1e-13)


def test_gaussian_exact_posterior_objective():
    x=np.array([-2.,.2,3.]);sd=.7
    for m in [0,20]:
        obj=ConditionalObjective(x,m,sd,"gaussian",fixed_nu=np.inf)
        p,_=obj.initial(0)
        expected=norm.logpdf(x,scale=np.sqrt(1+sd**2))
        np.testing.assert_allclose(obj(p,grad=False,individual=True),expected,atol=2e-13)


def test_direct_student_at_prior_has_zero_kl():
    nu,m,sd=5.,5,2.
    obj=ConditionalObjective(np.array([-.3,2.]),m,sd,"direct_student",fixed_nu=nu,
                             order=64,tau_order=192)
    p,_=obj.initial(0)
    p[:2]=0
    p[2:6]=np.log(covariance_scale(nu))
    p[6:]=np.log(nu-2)
    expected=-.5*(np.log(2*np.pi*sd**2)+(np.array([-.3,2.])**2+1)/sd**2)
    np.testing.assert_allclose(obj(p,grad=False,individual=True),expected,atol=2e-6)


def test_s3_two_starts_are_distinct():
    obj=ConditionalObjective(np.array([-.3,2.]),5,.7,"gaussian",fixed_nu=5.)
    p0,_=obj.initial(0);p1,_=obj.initial(1)
    assert not np.array_equal(p0,p1)


def test_full_covariance_laplace_matches_independent_radial_quadrature():
    common=dict(x=np.array([-.3,2.,4.]),m=5,sd=.7,family="full_covariance_gaussian")
    full=FullCovarianceObjective(**common,tau_order=256)
    diag=ConditionalObjective(**dict(common,family="gaussian_q_student_prior"),order=128)
    pf,_=full.initial(0);pd,_=diag.initial(0)
    np.testing.assert_allclose(full(pf,grad=False,individual=True),diag(pd,grad=False,individual=True),atol=3e-9)


def test_full_covariance_nonzero_correlation_gradients():
    obj=FullCovarianceObjective(np.array([-.3,2.]),5,.7,"full_covariance_gaussian",tau_order=128)
    p,_=obj.initial(1)
    value,grad=obj(p)
    assert np.isfinite(value) and np.isfinite(grad).all()
    for i in [0,1,2,7,8,9,len(p)-1]:
        h=1e-5;plus=p.copy();minus=p.copy();plus[i]+=h;minus[i]-=h
        expected=(obj(plus,grad=False)-obj(minus,grad=False))/(2*h)
        np.testing.assert_allclose(grad[i],expected,atol=3e-7,rtol=3e-6)


def test_independent_tail_reference_regression():
    # Actual fresh seed1203 extreme, whose raw density is about3.6e-15.
    x=-13.314892930195745
    expected=evidence([x],30.,.5,mu=1.,order=512)[0]
    np.testing.assert_allclose(independent_evidence(x,30.,.5,mu=1.),expected,atol=2e-11)


@pytest.mark.parametrize("nu",[2.1,3.,30.,200.])
def test_scaled_reference_extreme_densities(nu):
    for x in [-100.,0.,100.]:
        np.testing.assert_allclose(independent_evidence(x,nu,.5),evidence([x],nu,.5,order=512)[0],atol=2e-9)


@pytest.mark.parametrize("family",["gaussian","direct_student"])
@pytest.mark.parametrize("nu",[2.1,3.,30.])
def test_laplace_univariate_expectation_against_adaptive_density_integration(family,nu):
    x,mean,var,sd,qnu=1.,.5,2.,2.,3.
    obj=ConditionalObjective(np.array([x]),0,sd,family,fixed_nu=nu,
                             quadrature="laplace",laplace_order=256,tau_order=192)
    p=np.array([mean,np.log(var)]+([np.log(qnu-2)] if family=="direct_student" else []))
    if family=="direct_student":
        density=lambda z:t.pdf(z,qnu,loc=mean,scale=np.sqrt(var))
        entropy=t.entropy(qnu,loc=mean,scale=np.sqrt(var))
        expected_variance=var*qnu/(qnu-2)
    else:
        density=lambda z:norm.pdf(z,loc=mean,scale=np.sqrt(var))
        entropy=norm.entropy(scale=np.sqrt(var));expected_variance=var
    elogprior=quad(lambda z:t.logpdf(z,nu,scale=np.sqrt(covariance_scale(nu)))*density(z),
                   -np.inf,np.inf,epsabs=1e-11,epsrel=1e-11,limit=400)[0]
    rec=-.5*(np.log(2*np.pi*sd**2)+((x-mean)**2+expected_variance)/sd**2)
    got=obj(p,grad=False,individual=True)[0]
    np.testing.assert_allclose(got,rec+entropy+elogprior,atol=2e-8)
    assert obj.laplace_lower_tail_bound(p)<1e-10


@pytest.mark.parametrize("family",["gaussian_q_student_prior","student_q_student_prior"])
@pytest.mark.parametrize("m",[0,20])
def test_laplace_kernel_gradients(family,m):
    obj=ConditionalObjective(np.array([-.3,2.]),m,2.,family,quadrature="laplace",
                             laplace_order=192,tau_order=128)
    p,_=obj.initial(0);value,grad=obj(p)
    assert np.isfinite(value) and np.isfinite(grad).all()
    for i in [0,1,3,len(p)-1]:
        h=1e-5;plus=p.copy();minus=p.copy();plus[i]+=h;minus[i]-=h
        numerical=(obj(plus,grad=False)-obj(minus,grad=False))/(2*h)
        np.testing.assert_allclose(grad[i],numerical,atol=3e-7,rtol=3e-6)


def test_laplace_resolves_broad_gaussian_near_t_center():
    obj=ConditionalObjective(np.array([0.]),0,2.,"gaussian",fixed_nu=3.,
                             quadrature="laplace",laplace_order=192)
    mean=torch.tensor([0.],dtype=torch.float64);variance=torch.tensor([4.],dtype=torch.float64)
    got=obj.laplace_elog(mean,variance,variance,torch.tensor(3.,dtype=torch.float64)).item()
    expected=quad(lambda z:norm.pdf(z,scale=2)*np.log1p(z*z),-np.inf,np.inf,epsabs=1e-12)[0]
    np.testing.assert_allclose(got,expected,atol=2e-11)
