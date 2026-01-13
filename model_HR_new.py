# model_HR.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from constants import dreal, dcomplex


class AdSBHNet(nn.Module):
    """
    Metric-learning model for V(L) from lattice data.

    conventions (matches Mathematica aNetPT/bNetPT):
      a(z) = sum_{i=1}^N a_i z^i
      b(z) = sum_{i=1}^N b_i z^i
      f(z) = (1 - z^4) * exp(a(z))
      g(z) = exp(b(z)) / (1 - z^4)
    """

    def __init__(self, N: int = 5, std: float = 1.0):
        super().__init__()

        # deformation coefficients (PT polynomials)
        self.a = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dreal))
        self.b = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dreal))

        # overall vertical scale for V (R^2 / (2π α')) stored in log-space
        self.logcoef = nn.Parameter(torch.normal(0.0, std, size=(1,), dtype=dreal)[0])

        # additive shift to match the UV-normalised lattice potential
        self.shift = nn.Parameter(torch.tensor(0.0, dtype=dreal))

        # optional caches for curve sampling (not required by forward)
        self.curve_L = []
        self.curve_zs = []

    #  helpers 

    @staticmethod
    def _pos(x: torch.Tensor, eps: float = 1e-12, beta: float = 6.0) -> torch.Tensor:
        """Smooth positive guard that stays differentiable."""
        return eps + F.softplus(x - eps, beta=beta)

    def _to_tensor_like_a(self, z):
        """Cast z to a tensor on model device; preserve real/complex dtype."""
        if isinstance(z, torch.Tensor):
            return z.to(device=self.a.device)
        dt = dcomplex if isinstance(z, complex) else dreal
        return torch.as_tensor(z, dtype=dt, device=self.a.device)

    #  a(z), b(z) and their derivatives 

    # a(z) = sum_{i=1}^N a_i z^i  (self.a[0] multiplies z^1)
    def eval_a(self, z):
        z = self._to_tensor_like_a(z)
        out = torch.zeros_like(z, dtype=z.dtype)
        p = z  # z^1
        for c in self.a:
            out = out + c.to(z.dtype) * p
            p = p * z
        return out

    # a'(z) = sum_{i=1}^N i * a_i z^{i-1}
    def eval_da(self, z):
        z = self._to_tensor_like_a(z)
        out = torch.zeros_like(z, dtype=z.dtype)
        p = torch.ones_like(z, dtype=z.dtype)  # z^0
        for i, c in enumerate(self.a, start=1):
            out = out + (i * c.to(z.dtype)) * p
            p = p * z
        return out

    # b(z) = sum_{i=1}^N b_i z^i
    def eval_b(self, z):
        z = self._to_tensor_like_a(z)
        out = torch.zeros_like(z, dtype=z.dtype)
        p = z  # z^1
        for c in self.b:
            out = out + c.to(z.dtype) * p
            p = p * z
        return out

    # b'(z) = sum_{i=1}^N i * b_i z^{i-1}
    def eval_db(self, z):
        z = self._to_tensor_like_a(z)
        out = torch.zeros_like(z, dtype=z.dtype)
        p = torch.ones_like(z, dtype=z.dtype)  # z^0
        for i, c in enumerate(self.b, start=1):
            out = out + (i * c.to(z.dtype)) * p
            p = p * z
        return out

    #  metric functions f and g and their derivatives

    # f(z) = (1 - z^4) * exp(a(z))
    def eval_f(self, z):
        z = self._to_tensor_like_a(z)
        a = self.eval_a(z)
        return (1 - z**4) * torch.exp(a)

    # f'(z) = exp(a)(-4 z^3) + f * a'(z)
    def eval_df(self, z):
        z  = self._to_tensor_like_a(z)
        a  = self.eval_a(z)
        ap = self.eval_da(z)
        ea = torch.exp(a)
        f  = (1 - z**4) * ea
        return ea * (-4 * z**3) + f * ap

    # g(z) = exp(b(z)) / (1 - z^4)
    def eval_g(self, z):
        z = self._to_tensor_like_a(z)
        b = self.eval_b(z)
        return torch.exp(b) / (1 - z**4)

    # g'(z) = g(z) * [ b'(z) + 4 z^3 / (1 - z^4) ]
    def eval_dg(self, z):
        z  = self._to_tensor_like_a(z)
        g  = self.eval_g(z)
        bp = self.eval_db(z)
        return g * (bp + 4 * z**3 / (1 - z**4))

    #  core integrals (complex-safe) 

    def integrate_L(self, zs, Nu: int = 2000, eps: float = 1e-6):
        """
        Real/complex continuation of
          L(z_*) = (2/π) ∫_0^{z_*} dz  sqrt(g(z)) / sqrt( (z_*^4 f(z_*)) / (z^4 f(z)) - 1 )

        Mathematica version:
          computeLnetz[as, bs, zs] := 2/π * NIntegrate[
              Sqrt[gNet[as, bs, z]] / Sqrt[(zs^4 fNet[as, zs])/(z^4 fNet[as, z]) - 1],
              {z, 0, zs}
          ]

        Implemented with change of variables z = u * z_*, u ∈ (0,1).
        """
        device = self.a.device
        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(zs, dtype=dcomplex, device=device)
        zs = zs.to(dcomplex).view(-1)  # (B,)

        u = torch.linspace(eps, 1.0 - eps, Nu, dtype=dreal, device=device)  # (Nu,)
        du = torch.gradient(u)[0].to(dcomplex)

        zs_b = zs.view(-1, 1).to(dcomplex)  # (B,1)
        u_b  = u.view(1, -1).to(dcomplex)   # (1,Nu)

        z   = zs_b * u_b                   # (B,Nu)
        fz  = self.eval_f(z).to(dcomplex)
        gz  = self.eval_g(z).to(dcomplex)
        fzs = self.eval_f(zs_b).to(dcomplex)

        # Small complex regulator to avoid NaNs when the argument crosses zero
        epsc = torch.tensor(eps, dtype=dcomplex, device=device) * (1 + 1j)
        # Mathematica: sqrt((zs^4 f(z))/(z^4 f(zs)) - 1)
        denom = (zs_b**4 * fz) / (z**4 * fzs + epsc) - 1.0 + epsc
        integrand = torch.sqrt(gz) / torch.sqrt(denom)

        I  = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * (zs_b * du[1:])
        Iy = I.sum(dim=-1)

        L  = (2.0 / math.pi) * Iy
        return L  # complex
    

    def integrate_V(self, zs, Nu: int = 1500, eps: float = 1e-6):
        """
        Real/complex continuation of
          V(z_*) = 2π ( ∫_{eps}^{z_*} dz [ 1/z^2 ( sqrt(f g)/sqrt(1 - (z^4 f(z_*))/(z_*^4 f(z))) - 1 ) ] - 1/z_* )

        Mathematica version:
          computeVnetz[as, bs, zs] := 2π * (
              NIntegrate[
                  1/z^2 * (Sqrt[fNet[as, z] * gNet[as, bs, z]] /
                           Sqrt[1 - (z^4 fNet[as, zs])/(zs^4 fNet[as, z])] - 1),
                  {z, 10^-2, zs}
              ] - 1/zs
          )

        Implemented with z = u * z_*, u ∈ (0,1).
        """
        device = self.a.device

        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(zs, dtype=dcomplex, device=device)
        zs = zs.to(dcomplex).view(-1)                                     # (B,)

        # Start integration from eps (like Mathematica 10^-2), not 0
        u  = torch.linspace(eps, 1.0 - eps, Nu, dtype=dreal, device=device)
        du = torch.gradient(u)[0].to(dcomplex)
        zs_b = zs.view(-1, 1).to(dcomplex)                                # (B,1)
        u_b  = u.view(1, -1).to(dcomplex)                                 # (1,Nu)

        z   = zs_b * u_b                                                  # (B,Nu)
        fz  = self.eval_f(z).to(dcomplex)
        gz  = self.eval_g(z).to(dcomplex)
        fzs = self.eval_f(zs_b).to(dcomplex)

        epsc  = torch.tensor(eps, dtype=dcomplex, device=device) * (1 + 1j)
        inner = 1.0 - (z**4 * fzs) / (zs_b**4 * fz + epsc) + epsc         # (B,Nu)

        fg_sqrt = torch.sqrt(fz * gz)                                     # (B,Nu)
        term = fg_sqrt / torch.sqrt(inner) - 1.0
        integrand = term / (z**2 + epsc)                                  # (B,Nu)

        I_conn = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * (zs_b * du[1:])
        V_conn = I_conn.sum(dim=-1)                                       # (B,)
        V_conn = 2.0 * math.pi * V_conn

        V_disc = 2.0 * math.pi * (1.0 / zs_b.squeeze(-1))

        # Return raw integral WITHOUT logcoef scaling (to match Mathematica)
        # logcoef and shift should be applied externally
        return (V_conn - V_disc)




    #  optional: dL/dz_* (for L_max on real branch) 

    def integrate_dL(self, zs):
        """
        d/dz_* of T*L; scalar input recommended. Used by get_L_max() to locate the turning point on the real branch.
        This mirrors your original definition (kept for compatibility).
        """
        zs = zs if isinstance(zs, torch.Tensor) else torch.as_tensor(zs, dtype=dcomplex, device=self.a.device)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dreal, device=self.a.device)
        z = zs * (1 - y) * (1 + y)

        f = self.eval_f(z)
        fs = self.eval_f(zs.unsqueeze(-1))
        g = self.eval_g(z)

        dlogf  = -4 * z**3 / ((1 - z) * (1 + z) * (1 + z**2)) - self.eval_da(z)
        dlogfs = -4 * zs**3 / ((1 - zs) * (1 + zs) * (1 + zs**2)) - self.eval_da(zs)
        dlogg  =  4 * z**3 / ((1 - z) * (1 + z) * (1 + z**2)) + self.eval_db(z)

        f_over_fs = torch.exp(self.eval_a(zs) - self.eval_a(z)) \
                    * (1 - z) * (1 + z) * (1 + z**2) / ((1 - zs) * (1 + zs) * (1 + zs**2))

        integrand = (-4
                     - 2 * z * dlogg
                     + 4 * zs**4 * f_over_fs / z**4
                     - 2 * zs**4 * dlogf  * f_over_fs / z**3
                     + 2 * zs**5 * dlogfs * f_over_fs / z**4
                     + 2 * zs**4 * dlogg  * f_over_fs / z**3)

        denom = (zs**4 * f / (z**4 * fs) - 1) ** 1.5
        integrand = integrand / denom * y * torch.sqrt(g)

        # y=0 extrapolation and y=1 endpoint (as in your original)
        y = torch.cat((torch.tensor([0.0], dtype=dreal, device=self.a.device), y))
        integrand0 = ((integrand[1] - integrand[0]) / (y[2] - y[1]) * (-y[1]) + integrand[1]).unsqueeze(-1)
        integrand = torch.cat((integrand0, integrand))
        y = torch.cat((y, torch.tensor([1.0], dtype=dreal, device=self.a.device)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex, device=self.a.device)))

        dL = torch.trapz(integrand, y) / np.pi
        assert not torch.isnan(dL), f'integrate_dL({zs}) = {dL} for a = {self.a} b = {self.b}'
        return dL

    def get_L_max(self):
        """
        Find the last point on the real-z_* branch where L is maximal (turning point).
        """
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
        assert torch.abs(L_max.imag) < 1e-8
        return torch.tensor(zs_mid, dtype=dcomplex, device=self.a.device), L_max.real

    #  Newton-free interpolator & forward 

    def V_of_L_interp(self, L_query, zs_num: int = 2000, q: float = 2.5, eps: float = 1e-12):
        """
        Differentiable V(L) via parametric sampling:
          - power-law z_* grid dense near UV,
          - local quadratic in L for small interpolation bias.
        Returns REAL tensor (no shift).
        """
        device = self.a.device
        Lq = torch.as_tensor(L_query, dtype=dreal, device=device).reshape(-1)

        # dense near UV
        zmin, zmax = 0.02, 0.995
        u = torch.linspace(0.0, 1.0, zs_num, dtype=dreal, device=device)
        zs = zmin + (zmax - zmin) * u**q  # (Nc,)

        # parametric curve
        Lc = self.integrate_L(zs).real
        Vc = self.integrate_V(zs).real

        # sort by L and fit locally (quadratic)
        idx = torch.argsort(Lc)
        Lc, Vc = Lc[idx], Vc[idx]
        Ld = Lc.detach()

        pos = torch.searchsorted(Ld, Lq, right=True)
        i = pos.clamp(1, Ld.numel() - 2)
        i0, i1, i2 = i - 1, i, i + 1

        x0, x1, x2 = Ld[i0], Ld[i1], Ld[i2]
        y0, y1, y2 = Vc[i0], Vc[i1], Vc[i2]

        ones = torch.ones_like(x0)
        A = torch.stack([
            torch.stack([x0 * x0, x0, ones], dim=-1),
            torch.stack([x1 * x1, x1, ones], dim=-1),
            torch.stack([x2 * x2, x2, ones], dim=-1)
        ], dim=1)  # (Nq,3,3)
        Y = torch.stack([y0, y1, y2], dim=-1)[..., None]  # (Nq,3,1)
        coeff = torch.linalg.solve(A, Y).squeeze(-1)      # (Nq,3)
        qa, qb, qc = coeff[:, 0], coeff[:, 1], coeff[:, 2]

        V_quad = qa * Lq * Lq + qb * Lq + qc

        # linear extrapolation outside support
        left  = (Lq <= Ld[0])
        right = (Lq >= Ld[-1])
        mL = (Vc[1] - Vc[0])   / (Ld[1] - Ld[0]   + eps)
        mR = (Vc[-1] - Vc[-2]) / (Ld[-1] - Ld[-2] + eps)
        V_left  = Vc[0]  + mL * (Lq - Ld[0])
        V_right = Vc[-1] + mR * (Lq - Ld[-1])

        V_core = torch.where(left, V_left, V_quad)
        V_core = torch.where(right, V_right, V_core)
        return V_core

    def forward(self, Ls):
        """
        Default forward: compute V(L) via differentiable interpolation and add the learned shift.
        Returns complex dtype for backward compatibility (imag=0).
        """
        V_core = self.V_of_L_interp(Ls).real
        V = V_core + self.shift
        return torch.complex(V, torch.zeros_like(V, dtype=dreal))
