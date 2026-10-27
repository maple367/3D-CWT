"""Reciprocity is transpose symmetry, not geometric or Hermitian symmetry."""
from unittest.mock import patch
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import linear_sum_assignment
from cwt3d.greens import HomogeneousGreen, LayeredGreen, PreparedLayeredGreen, PoleError
from cwt3d.quadrature import green_overlap, interval_rule
from cwt3d.waveguide import TEWaveguide
from cwt3d.geometry import CircleHole, RightTriangleHole
from cwt3d.coupling import build_coupling


class ComplexField:
    def theta(self, z):
        z = np.asarray(z)
        return (1 + .19*z) * np.exp((.17 + 2.3j)*z)


def two_triangle_reference(mode, receiver, source, kernel, order, *, reciprocity=True):
    """Frozen original expression, with two independent kernel calls."""
    z,w=interval_rule(receiver,order);s,v=interval_rule(source,order)
    if tuple(receiver)!=tuple(source):
        return np.sum(w[:,None]*v[None,:]*mode.theta(z)[:,None].conj()*kernel(z[:,None],s[None,:])*mode.theta(s)[None,:])
    u,w=interval_rule((0.,1.),order);a,b=receiver
    x=a+(b-a)*u[:,None];y=a+(b-a)*u[:,None]*u[None,:]
    weights=(b-a)**2*u[:,None]*w[:,None]*w[None,:]
    return np.sum(weights*(mode.theta(x).conj()*kernel(x,y)*mode.theta(y)+mode.theta(y).conj()*kernel(y,x)*mode.theta(x)))


@pytest.mark.parametrize('cls',[HomogeneousGreen,LayeredGreen,PreparedLayeredGreen])
def test_builtin_reciprocal_kernel_is_evaluated_once(cls,monkeypatch):
    if cls is HomogeneousGreen:
        kernel=cls(1.3,2.3-.2j,.7)
    else:
        kernel=cls(1.3,[1-.1j,5-.3j,2-.2j,1.6-.1j],[.3,.4],.7)
    calls=[];original=cls.__call__
    def counted(self,z,s):
        calls.append(1)
        return original(self,z,s)
    monkeypatch.setattr(cls,'__call__',counted)
    actual=green_overlap(ComplexField(),(.02,.27),(.02,.27),kernel,8)
    assert len(calls)==1
    expected=two_triangle_reference(ComplexField(),(.02,.27),(.02,.27),kernel,8)
    np.testing.assert_allclose(actual,expected,rtol=5e-12,atol=2e-14)


@pytest.mark.parametrize('cls',[LayeredGreen,PreparedLayeredGreen])
@pytest.mark.parametrize('eps,widths,k0,kp',[
    ([1,5,2,3,1.5],[.3,.4,.2],1.4,.6),
    ([1-.1j,5-.3j,2-.2j,3-.5j,1.5-.1j],[.3,.4,.2],1.4,1.7),
    ([2,1,3],[.7],1.,1.),
    ([1,2,1],[10000.],1.,3.),
])
def test_asymmetric_lossy_internal_cutoff_and_thick_layers(cls,eps,widths,k0,kp):
    kernel=cls(k0,eps,widths,kp)
    edges=np.r_[0,np.cumsum(widths)]
    for left,right in zip(edges[:-1],edges[1:]):
        interval=(left+.03,min(right-.03,left+.25))
        got=green_overlap(ComplexField(),interval,interval,kernel,12)
        old=two_triangle_reference(ComplexField(),interval,interval,kernel,12)
        np.testing.assert_allclose(got,old,rtol=5e-12,atol=2e-14)


@pytest.mark.parametrize('cls',[LayeredGreen,PreparedLayeredGreen])
def test_nonzero_near_guided_pole_and_pole_rejection(cls):
    guide=TEWaveguide([1,9,1],[.6],k0=4.)
    beta=guide.solve_mode().beta
    for offset in (1e-4,1e-6):
        kernel=cls(4.,guide.eps,guide.thickness_um,beta+offset)
        got=green_overlap(ComplexField(),(.01,.59),(.01,.59),kernel,14)
        old=two_triangle_reference(ComplexField(),(.01,.59),(.01,.59),kernel,14)
        np.testing.assert_allclose(got,old,rtol=5e-12,atol=2e-13)
    with pytest.raises(PoleError):
        kernel=cls(4.,guide.eps,guide.thickness_um,beta)
        green_overlap(ComplexField(),(.01,.59),(.01,.59),kernel,8)


def test_independent_adaptive_complex_field_integral():
    mode=ComplexField();kernel=HomogeneousGreen(1.3,2.3-.2j,.7)
    left,right=.03,.57
    def integrate_complex(f,a,b):
        if a==b:return 0j
        return quad(lambda x:f(x).real,a,b,epsabs=1e-12,epsrel=1e-12)[0]+1j*quad(lambda x:f(x).imag,a,b,epsabs=1e-12,epsrel=1e-12)[0]
    def outer(z):
        f=lambda s:mode.theta(z).conj()*kernel(z,s)*mode.theta(s)
        return integrate_complex(f,left,z)+integrate_complex(f,z,right)
    expected=integrate_complex(outer,left,right)
    actual=green_overlap(mode,(left,right),(left,right),kernel,16)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)


def test_unknown_nonreciprocal_callable_keeps_original_path():
    calls=[]
    def kernel(z,s):
        calls.append(1)
        return 1+.3j*z+.8*s
    got=green_overlap(ComplexField(),(.02,.38),(.02,.38),kernel,10)
    assert len(calls)==2
    old=two_triangle_reference(ComplexField(),(.02,.38),(.02,.38),kernel,10)
    np.testing.assert_array_equal(got,old)


@pytest.mark.parametrize('field,k0,kp,bounds',[
    (1e160,1e150,3e150,(0.,1e-150)),
    (1e-170,1e-150,3e-150,(0.,1.)),
])
def test_reuse_preserves_safe_multiplication_order(field,k0,kp,bounds):
    class ConstantField:
        def theta(self,z):
            return np.full_like(z,field,dtype=complex)
    kernel=HomogeneousGreen(k0,1.,kp)
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        old=two_triangle_reference(ConstantField(),bounds,bounds,kernel,10)
        got=green_overlap(ConstantField(),bounds,bounds,kernel,10)
    assert np.isfinite(old) and old!=0
    np.testing.assert_allclose(got,old,rtol=5e-13,atol=0)


def test_subclass_overriding_kernel_is_not_assumed_reciprocal():
    calls=[]
    class Nonreciprocal(LayeredGreen):
        def __call__(self,z,s):
            calls.append(1)
            return super().__call__(z,s)+.7*(np.asarray(z)-np.asarray(s))
    kernel=Nonreciprocal(1.3,[1,5,1.6],[.4],.7)
    got=green_overlap(ComplexField(),(.02,.38),(.02,.38),kernel,10)
    assert len(calls)==2
    old=two_triangle_reference(ComplexField(),(.02,.38),(.02,.38),kernel,10)
    np.testing.assert_array_equal(got,old)


@pytest.mark.parametrize('cls',[LayeredGreen,PreparedLayeredGreen])
def test_distinct_intervals_unchanged(cls):
    kernel=cls(1.3,[1-.1j,5-.3j,2-.2j,1.6-.1j],[.3,.4],.7)
    left,right=(.02,.27),(.34,.65)
    got=green_overlap(ComplexField(),left,right,kernel,10)
    old=two_triangle_reference(ComplexField(),left,right,kernel,10)
    np.testing.assert_array_equal(got,old)


@pytest.mark.parametrize('backend',['reference','prepared'])
def test_two_distinct_patterned_layers_without_geometric_symmetry(backend):
    guide=TEWaveguide([1.,8.,10.,2.25],[.13,.19],k0=6.)
    mode=guide.solve_mode();a=2*np.pi/mode.beta
    f=np.pi*.13**2
    first=CircleHole(.13*a,(a,a),(.36*a,.47*a),8+2*f,8-2*(1-f))
    f=.35*.27/2
    second=RightTriangleHole((.35*a,.27*a),(a,a),(.63*a,.54*a),10+2*f,10-2*(1-f))
    kwargs=dict(cutoff=2,quadrature_order=10,fourier_order=20,backend=backend)
    with patch('cwt3d.coupling.green_overlap',two_triangle_reference):
        old=build_coupling(mode,a,{1:first,2:second},**kwargs)
    new=build_coupling(mode,a,{1:first,2:second},**kwargs)
    for name in ('C1D','Crad','C2D','C2D_green','C2D_local','C','I','J','K','xi','mu','nu','varsigma'):
        np.testing.assert_allclose(getattr(new,name),getattr(old,name),rtol=5e-12,atol=2e-13,err_msg=name)
    before,after=np.linalg.eigvals(old.C),np.linalg.eigvals(new.C)
    i,j=linear_sum_assignment(abs(before[:,None]-after[None,:]))
    np.testing.assert_allclose(before[i],after[j],rtol=5e-12,atol=2e-13)
