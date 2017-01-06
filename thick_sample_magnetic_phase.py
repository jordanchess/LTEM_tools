"""Calculate the magnetic phase shift of each voxel in a 3D object.

Tool to calculate the LTEM phase shift of an arbitrary shaped arbitrary
magnetization sample using the approach of Humphrey et al.
Humphrey, E., and M. De Graef. "On the computation of the magnetic phase shift
for magnetic nano-particles of arbitrary shape using a spherical projection
model." Ultramicroscopy 129 (2013): 36-41
"""

import numpy as np
from numba import jit  # , vectorize, complex128
import numexpr as ne
import math
# from threading import Thread
# from threading import Lock
# from multiprocessing import cpu_count


def kx_ky_k_perp(pxsize, x_len, y_len):
    """
    Create the Fourier space coordinate system.

    :param pxsize:    pixel width assumed to be square pixels use nm to be
                      consistent with other functions in this module
    :param x_len, y_len:  width and height of the real-space "image" in units
                          of pixels
    :return (kx, ky, k_perp):   kx array with dimensions [y_len, x_len]
                                ky array with dimensions [y_len, x_len]
                                k_perp array with dimensions [y_len, x_len]
                                k_perp = sqrt(kx^2+ky^2)

    """
    kx, ky = np.meshgrid(np.fft.fftfreq(y_len, d=pxsize),
                         np.fft.fftfreq(x_len, d=pxsize))
    k_perp = np.sqrt((kx**2) + (ky**2))
    return kx, ky, k_perp


def S_x_S_y(kx, ky, k_perp, R, Ms):
    """Create the shape functions S_x and S_y of a sphere.

    :param (kx, ky, k_perp):   kx array with dimensions [y_len, x_len]
                               ky array with dimensions [y_len, x_len]
                               k_perp array with dimensions [y_len, x_len]
                               k_perp = sqrt(kx^2+ky^2)
    :param Ms:  Saturation magnetization should be in A/nm
    :param R:  Radius of the sphere should be in nm
    :return S_x, S_y:
    """
    constant = 0.6071  # 1/(A nm) \mu_0/\Phi_0
    c = 4*(np.pi**2)*(R**2)*1.0j*Ms*constant
    kR = k_perp*R
    Sx = (((np.sin(kR)/(kR**2)-np.cos(kR)/(kR)))/(k_perp**3))*kx
    Sy = (((np.sin(kR)/(kR**2)-np.cos(kR)/(kR)))/(k_perp**3))*ky
    Sx[np.isnan(Sx)] = 0.0
    Sy[np.isnan(Sy)] = 0.0
    return c*Sx, c*Sy


def ne_phi_mk_delta_ijk(kx, ky, Sx, Sy, mu_x, mu_y, mu_z, a, cos_d,
                        sin_d, i, j, k, x0, y0, z0):
    """Calculate the phase shift for single sphere.

    :param (kx, ky, k_perp):   kx array with dimensions [y_len, x_len]
                               ky array with dimensions [y_len, x_len]
                               k_perp array with dimensions [y_len, x_len]
                               k_perp = sqrt(kx^2+ky^2)
    :param (Sx, Sy):   As defined in the paper constructed using the function
                       S_x_S_y()
    :param mu_x, mu_y, mu_z:  x, y, z components of the normalized
                              magnetization
    :param a:  lattice parameter of the grid representing the sample
    :param (cos_d, sin_d):  sine and cosine of the sample angle delta
    :param (i,j,k):  x, y, z indices
    :param x_shift:  position of axis of rotation
    :return phi_ijk:  phase shift from one sphere
    """
    trig = cos_d*(i-x0) + sin_d*(k-z0)
    sy_cof = mu_x[i, j, k]*cos_d + mu_z[i, j, k]*sin_d
    sx_cof = mu_y[i, j, k]
    return -ne.evaluate('exp(2.0*a*1.0j*(kx*trig+ky*(j-y0)))*(sy_cof*Sy-sx_cof*Sx)')


# this is the best optimized I have been able to get so far
@jit
def phi_mk_delta(kx, ky, Sx, Sy, mu_x, mu_y, mu_z, a,
                 delta, i_s, j_s, k_s, phi, x0, y0, z0):
    """Calculate the phase shift for all the spheres in sample.

    :param (kx, ky, k_perp):   kx array with dimensions [y_len, x_len]
                               ky array with dimensions [y_len, x_len]
                               k_perp array with dimensions [y_len, x_len]
                               k_perp = sqrt(kx^2+ky^2)
    :param (Sx, Sy):   As defined in the paper constructed using the function
                       S_x_S_y()
    :param mu_x, mu_y, mu_z:  x, y, z components of the normalized
                              magnetization
    :param a:  lattice parameter of the grid representing the sample
    :param delta:  sample tilt angle delta about the y-axis
    :param (i,j,k):  x, y, z indices for each sphere
    :param (x0,y0,z0):  x, y, z position of the center of rotation, seems to
                        make the most sense that is is the center of mass but
                        it can be any point
    :return phi_ijk:  phase shift from one sphere
    """
    sin_d = math.sin(delta)
    cos_d = math.cos(delta)
    for coord in zip(i_s, j_s, k_s):
        i, j, k = coord
        phi += ne_phi_mk_delta_ijk(kx, ky, Sx, Sy, mu_x, mu_y, mu_z, a,
                                   cos_d, sin_d, i, j, k, x0, y0, z0)
    return phi
