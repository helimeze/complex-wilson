import logging
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from constants import dreal, dcomplex


class AdSBHNet(nn.Module):
    def __init__(self, N=5, std=1.0):
        super(AdSBHNet, self).__init__()
        self.a = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dreal)) 
        self.b = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dreal))
        '''
        `self.logcoef` is the log of the dimensionless parameter
        R^2/(2*pi*alpha') which multiplies the static potential V.
        '''
        #previously self.logcoef = nn.Parameter(torch.normal(0.0, std, size=(1,), dtype=dreal)[0])
        self.logcoef = nn.Parameter(torch.normal(0.0, std, size=(1,), dtype=dreal)[0])#nn.Parameter(torch.tensor(0.15089977670036966,dtype=dreal).log())

        '''
        The lattice data is supposed to be shifted such that it behaves
        correctly in the UV. `self.shift` holds that parameter.
        '''
        self.shift = nn.Parameter(torch.tensor(0.0, dtype=dreal))
        self.curve_L = []
        self.curvs_zs = []

    def forward(self, Ls):
        '''
        Initial version with torch.trapz
        instead of torchdiffeq.
        '''
        V = torch.zeros_like(Ls, dtype=dcomplex)

        self.find_curve()
        curve = interp1d(self.curve_L, self.curve_zs)
        zs_max, L_max = self.get_L_max()

        for i, L in enumerate(Ls):
            init = complex(curve(L.item()))
            _zs = self.find_zs_newton(L, init)
            # TODO: Verify that flipping possibly negative imaginary parts is always safe
            zs = torch.complex(_zs.real, _zs.imag.abs())
            assert zs.real > 0, f'Real part of zs is negative: {zs} for L = {L}'
            V[i] = self.integrate_V(zs)
            assert not torch.isnan(V[i])
        return V

    def find_curve(self):
        zs_max, L_max = self.get_L_max()
        self.curve_L = [0.0, L_max.item()]
        self.curve_zs = [complex(0.0, 0.0), zs_max.item()]

        L_step_default = (1 - L_max.item()) / 50
        L_step = L_step_default
        while True:
            L = self.curve_L[-1] + L_step
            if np.abs(self.curve_zs[-1].imag) < 1e-8:
                init = self.curve_zs[-1] + 0.1j
            else:
                init = self.curve_zs[-1]
            _zs = self.find_zs_newton(L, init)
            zs = complex(_zs.real.item(), _zs.imag.abs().item() if L > L_max else 0.0)
            if np.abs(zs - self.curve_zs[-1]) > 0.1:
                L_step /= 2
            else:
                self.curve_zs.append(zs)
                self.curve_L.append(L)
                L_step = L_step_default
            if self.curve_L[-1] > 1:
                break

    def find_zs_newton(self, L, init, max_steps=25, retry=True):
        if not isinstance(init, torch.Tensor) or init.dtype != dcomplex:
            init = torch.as_tensor(init, dtype=dcomplex)
        zs = [init]
        _L = self.integrate_L(zs[-1])
        for i in range(max_steps):
            dL = self.integrate_dL(zs[-1])
            zs.append(zs[-1] - (_L - L) / dL)
            if zs[-1].abs() > 100 or torch.isnan(zs[-1]):
                logging.error(f'Something wrong in Newton:\n\tzs = {[_zs.item() for _zs in zs]}\n\tL = {L}\n\t_L = {_L}\n\tdL = {dL}')
            _L = self.integrate_L(zs[-1])
            diff = torch.abs(_L - L)
            if diff < 1e-8:
                return zs[-1]
        if retry:
            logging.warning(f'Newton\'s method failed to converge in {max_steps} iterations for L = {L}\n\tzs = {zs[-1]}\n\tdiff = {diff}\n\tinit = {init}. Retrying with default init 0.5+0.5j.')
            return self.find_zs_newton(L, 0.5 + 0.5j, retry=False)
        else:
            logging.error(f'Newton\'s method failed to converge in {max_steps} iterations for L = {L}\n\tzs = {zs[-1]}\n\tdiff = {diff}\n\tinit = {init}.')
            return zs[-1]

    def get_L_max(self):
        '''
        Returns the point where L is maximal such that
        zs is still real. This is the last point on the
        real axis along the real L curve.
        '''
        zs_UV, zs_IR = 0.001, 0.999
        dL_IR = self.integrate_dL(zs_IR).real
        dL_UV = self.integrate_dL(zs_UV).real
        assert dL_IR < 0 and dL_UV > 0
        while zs_IR - zs_UV > 1e-8:
            zs_mid = (zs_UV + zs_IR) / 2
            dL_mid = self.integrate_dL(zs_mid).real
            if dL_mid < 0:
                zs_IR = zs_mid
            else:
                zs_UV = zs_mid
        zs_mid = (zs_UV + zs_IR) / 2
        L_max = self.integrate_L(zs_mid)
        assert L_max.imag.abs() < 1e-8
        return torch.tensor(zs_mid, dtype=dcomplex), L_max.real

    def integrate_L(self, zs):
        '''
        This computes the dimensionless combination T*L,
        where T = 1/(pi*z_h).
        '''
        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(
            zs, dtype=dcomplex)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dreal)
        z = zs * (1 - y) * (1 + y)
        sqrtg = self.eval_g(z).sqrt()
        f_over_fs = torch.exp(self.eval_a(zs) - self.eval_a(z)) * (1 - z) * (1 + z) * (1 + z**2) \
            / ((1 - zs) * (1 + zs) * (1 + zs**2))
        integrand = sqrtg / \
            torch.sqrt(f_over_fs / ((1 - y)**4 * (1 + y)**4) - 1) * y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dreal)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex)))
        # Integrate
        L = 4 * zs * torch.trapz(integrand, y) / np.pi
        assert not torch.isnan(L), f'integrate_L({zs}) = {L} for a = {self.a} b = {self.b}'
        return L

    def integrate_dL(self, zs):
        '''
        This computes the derivative of T*L w.r.t. z_*/z_h.
        '''
        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(
            zs, dtype=dcomplex)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dreal)
        z = zs * (1 - y) * (1 + y)
        f = self.eval_f(z)
        fs = self.eval_f(zs.unsqueeze(-1))
        g = self.eval_g(z)
        dlogf = -4 * z**3 / ((1 - z) * (1 + z) * (1 + z**2)) - self.eval_da(z)
        dlogfs = -4 * zs**3 / ((1 - zs) * (1 + zs) * (1 + zs**2)) - self.eval_da(zs)
        dlogg = 4 * z**3 / ((1 - z) * (1 + z) * (1 + z**2)) + self.eval_db(z)
        f_over_fs = torch.exp(self.eval_a(zs) - self.eval_a(z)) * (1 - z) * (1 + z) * (1 + z**2) \
            / ((1 - zs) * (1 + zs) * (1 + zs**2))
        integrand = -4 - 2 * z * dlogg + 4 * zs**4 * f_over_fs / z**4 - 2 * zs**4 * dlogf \
            * f_over_fs / z**3 + 2 * zs**5 * dlogfs * f_over_fs / z**4 + 2 * zs**4 * dlogg \
            * f_over_fs / z**3
        integrand /= (zs**4 * f / (z**4 * fs) - 1)**1.5
        integrand *= y * g.sqrt()
        # Extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add known value for y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dreal)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex)))
        # Integrate
        dL = torch.trapz(integrand, y) / np.pi
        assert not torch.isnan(dL), f'integrate_dL({zs}) = {dL} for a = {self.a} b = {self.b}'
        return dL

    def integrate_V(self, zs):
        V_c = self.integrate_V_connected(zs)
        # Then we subtract the disconnected configuration
        V_d = self.integrate_V_disconnected(zs)
        return 2*(V_c - V_d)

    def integrate_V_connected(self, zs):
        '''
        This computes the connected contribution of V/T,
        where T = 1/(pi*z_h).
        '''
        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(
            zs, dtype=dcomplex)
        zsreal = zs.real.item()# if torch.is_complex(zs) else zs
        upperlimit = 1-0.005/(2.0*zsreal)#np.sqrt(1-10**(-5)/zsreal)
        y = torch.linspace(0.01, upperlimit, steps=1000, dtype=dreal)
        z = zs * (1-y**2)
        integrand = 2*y/(zs*(1-y**2)**2) *(torch.sqrt(self.eval_f(z)*self.eval_g(z)/(1-(1-y**2)**4*self.eval_f(zs)/self.eval_f(z)))-1)
        # We extrapolate to y=0
        #y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        #integrand = torch.cat(
        #    (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=0.999999
        #y = torch.cat((y, torch.tensor([1.0], dtype=dreal)))
        #integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex)))
        # Integrate
        coef = self.logcoef.exp()
        V = np.pi *coef* torch.trapz(integrand, y)  # * 2*coef / zs
        assert not torch.isnan(V), f'integrate_V_connected({zs}) = {V} for a = {self.a} b = {self.b}'
        return V

    def integrate_V_disconnected(self, zs):
        '''
        This computes the disconnected contribution of V/T,
        where T = 1/(pi*z_h).
        '''
        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(
            zs, dtype=dcomplex)
        # Coordinate is y = (1 - z) / (1 - zs)
        y = torch.linspace(0.001, 1, steps=1000, dtype=dreal)
        z = 1 - (1 - zs) * y
        fg = torch.exp(self.eval_b(z) - self.eval_a(z))
        # NOTE: This assumes that f(z)*g(z)->1 when z->1
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        #integrand = torch.cat((torch.tensor([1.0], dtype=dreal), integrand))
        coef = self.logcoef.exp()
        V = coef*np.pi /(zs**(1)) #-coef*np.pi #*2*coef in both
        assert not torch.isnan(V), f'integrate_V_disconnected({zs}) = {V} for a = {self.a} b = {self.b}'
        return V

    #def eval_a(self, z):
    #    z = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
    #    out = torch.zeros_like(z)
    #    for i, ci in enumerate(self.a):
    #        out += ci **z**(i + 1) / (i + 1) 
    #    # Normalize
    #    return out 
    
    def eval_a(self, z):
        z = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
        out = torch.zeros_like(z)
        for i, ci in enumerate(self.a):
            for j, cj in enumerate(self.a):
                out += ci * cj *z**(i + j + 3) / (i + j + 3) 
        # Normalize
        return out 
    
    def eval_da(self, z):
        out = torch.zeros_like(z)
        for i, ci in enumerate(self.a):
            for j, cj in enumerate(self.a):
                out += ci * cj * z**(i + j + 2)
        # Normalize
        return out

    #def eval_da(self, z):
    #    out = torch.zeros_like(z)
    #    for i, ci in enumerate(self.a):
    #        out += ci * z**(i)
    #    # Normalize
    #    return out

    def eval_f(self, z):
        z = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
        return (1-z**4) / self.eval_a(z).exp()

    def eval_df(self, z):
        return -self.eval_f(z) * self.eval_da(z) - 4 * z**3 / self.eval_a(z).exp()

    def eval_b(self, z):
        z = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
        out = torch.zeros_like(z)
        for i, ci in enumerate(self.b):
            out += ci * z**(i+1) 
        a1 = self.eval_a(torch.tensor(1.0, dtype=dreal))
        N = len(self.b)
        out -= (self.b.sum() - a1) * z**(N + 1)
        return out

    def eval_db(self, z):
        out = torch.zeros_like(z)
        for i, ci in enumerate(self.b):
            out += (i + 1) * ci * z**i
        a1 = self.eval_a(torch.tensor(1.0, dtype=dreal))
        N = len(self.b)
        out -= (N + 1) * (self.b.sum() - a1) * z**N
        return out

    def eval_g(self, z):
        z = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
        return self.eval_b(z).exp() / (1-z**4) 

    def eval_dg(self, z):
        return self.eval_g(z) * (4 * z**3 / (1-z**4) + self.eval_db(z))

