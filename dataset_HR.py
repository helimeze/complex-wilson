import numpy.random as npr
import numpy as np
from numpy import real, imag #prev. from scipy import real, imag
from scipy.integrate import quad
from torch.utils.data.dataset import Dataset
import torch
from constants import dreal, dcomplex


class AdSBHDataset(Dataset):
    def __init__(self, N=1000, threshold=0.01, coef=1.0, file=None): #previously coef = 1.0
        '''
        L, V, and sigma is understood to correspond to the
        dimensionless combinations T*L, V/T, and sigma/T.
        '''
        if file:
            data = np.loadtxt(file)
            self.L = torch.tensor(data[0], dtype=dreal)
            self.V = torch.tensor(data[1] + 1j * data[3], dtype=dcomplex)
            self.sigma = torch.tensor(data[2] + 1j * data[4], dtype=dcomplex)
            self.zs, self.coef = None, None#0.15089977670036966 # previously None, None
        else:
            self.coef = coef
            self.L, self.zs, self.V, self.sigma = self.gen_data(N, threshold)

    def __getitem__(self, index):
        return self.L[index], self.V[index], self.sigma[index]

    def __len__(self):
        return len(self.L)

    def gen_data(self, N, threshold):
        zs_max, L_max = self.get_L_max()
        L_data = []
        zs_data = []
        V_data = []
        sigma_data = []
        for _ in range(N):
            L = npr.uniform(0.01, 2 / np.pi)
            if L <= L_max:
                # Binary search
                zs = self.find_zs_binary(L, zs_max)
            else:
                # Newton's method
                # TODO: Add a smarter init for Newton
                zs = self.find_zs_newton(L, zs_max + 0.1j)
            V = self.integrate_V(zs) + npr.uniform(-threshold, threshold)
            L_data.append(L)
            zs_data.append(zs)
            V_data.append(V)
            sigma_data.append(npr.uniform(0.1, 0.2) * (1 + 1j))
        L_data = torch.tensor(L_data, dtype=dreal)
        zs_data = torch.tensor(zs_data, dtype=dcomplex)
        V_data = torch.tensor(V_data, dtype=dcomplex)
        sigma_data = torch.tensor(sigma_data, dtype=dcomplex)
        return L_data, zs_data, V_data, sigma_data

    def find_zs_binary(self, L, zs_max):
        '''
        Intended for real zs.
        '''
        zs_low = 0.001
        zs_high = zs_max
        while zs_high - zs_low > 1e-8:
            zs_mid = (zs_high + zs_low) / 2
            L_mid = self.integrate_L(zs_mid)
            assert np.abs(L_mid.imag) < 1e-8
            L_mid = L_mid.real
            if L_mid < L:
                zs_low = zs_mid
            else:
                zs_high = zs_mid
        zs_mid = (zs_high + zs_low) / 2
        return zs_mid

    def find_zs_newton(self, L, init, max_steps=10):
        zs = init
        for i in range(max_steps):
            zs -= (self.integrate_L(zs) - L) / self.integrate_dL(zs)
            if np.abs(self.integrate_L(zs) - L) < 1e-8:
                break
        return zs

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
        assert np.abs(L_max.imag) < 1e-8
        return zs_mid, L_max.real

    def eval_f(self, z):
        return (1 - z**4)

    def eval_df(self, z):
        return -4 * z**3

    def eval_g(self, z):
        return 1 / (1 - z**4)


    def integrate_L(self, zs):
        def integrand(y):
            z = zs * (1 - y) * (1 + y)
            return np.sqrt(self.eval_g(z)) * y / np.sqrt(
                self.eval_f(z) / ((1 - y)**4 * (1 + y)**4 * self.eval_f(zs)) - 1)

        def integrand_real(y):
            return real(integrand(y))

        def integrand_imag(y):
            return imag(integrand(y))

        L = 4 * zs * (quad(integrand_real, 0, 1)
                      [0] + 1j * quad(integrand_imag, 0, 1)[0]) / np.pi
        return L

    def integrate_dL(self, zs):
        def integrand(y):
            # NOTE: This only holds for AdS-BH
            return 4 * (1 - y)**2 * (1 + y)**2 * (1 - 3 * zs**4 + (1 - y)**4 * (1 + y)**4 * zs**4 * (1 + zs**4)) / \
                (np.sqrt((1 - zs) * (1 + zs) * (1 + zs**2)) * np.sqrt(4 - 6 * y**2 + 4 * y**4 - y**6) *
                    (1 - (1 - y)**4 * (1 + y)**4 * zs**4)**1.5)

        def integrand_real(y):
            return real(integrand(y))

        def integrand_imag(y):
            return imag(integrand(y))

        dL = quad(integrand_real, 0, 1)[0] + 1j * \
            quad(integrand_imag, 0, 1)[0] / np.pi
        return dL

    def integrate_V(self, zs):
        def integrand(y):
            z = zs * (1 - y**2)
            return 2*y/(zs*(1-y**2)**2) *(np.sqrt(self.eval_f(z)*self.eval_g(z)/(1-(1-y**2)**4*self.eval_f(zs)/self.eval_f(z)))-1)

        def integrand_real(y):
            return real(integrand(y))

        def integrand_imag(y):
            return imag(integrand(y))

        def disconnected(y):
            z = 1 - (1 - zs) * y
            # Simplified by noting that in AdS, f(z)*g(z) = 1.
            # NOTE: Only works for AdS.
            # return np.sqrt(self.eval_f(z) * self.eval_g(z)) / z**2
            return 1 / zs

        def disconnected_real(y):
            return real(disconnected(y))

        def disconnected_imag(y):
            return imag(disconnected(y))

        V = 2*np.pi *\
            (quad(integrand_real, 0, np.real(zs))[0] +
             1j * quad(integrand_imag, 0, np.real(zs))[0])
        V -= 2*np.pi * \
             (quad(disconnected_real, 0, 1)
                        [0] + 1j * quad(disconnected_imag, 0, 1)[0])
        return V
    
    