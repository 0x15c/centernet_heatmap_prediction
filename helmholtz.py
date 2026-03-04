import numpy as np

def helmholtz_hodge_2d_fft(F, dx=1.0, dy=1.0, return_potentials=False):
    """
    2D Helmholtz–Hodge decomposition on a periodic grid using FFT.

    Parameters
    ----------
    F : np.ndarray
        Vector field, shape (2, Ny, Nx) or (2, H, W).
        F[0]=u(x,y), F[1]=v(x,y).
    dx, dy : float
        Grid spacing in x and y. (Must be uniform; periodic BC assumed.)
    return_potentials : bool
        If True, also return Phi, Psi.

    Returns
    -------
    F_grad : np.ndarray
        Curl-free component ∇Phi, shape (2, Ny, Nx).
    F_rot : np.ndarray
        Divergence-free component ∇⊥Psi, shape (2, Ny, Nx).
    H : np.ndarray
        Harmonic/mean component, constant field, shape (2, Ny, Nx).
    (optional) Phi, Psi : np.ndarray
        Scalar potential and stream function, shape (Ny, Nx).
    """
    if F.shape[0] != 2:
        raise ValueError("F must have shape (2, Ny, Nx)")

    u = F[0]
    v = F[1]
    Ny, Nx = u.shape

    # FFTs
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)

    # Frequencies in cycles per unit length
    kx = np.fft.fftfreq(Nx, d=dx).reshape(1, Nx)   # (1, Nx)
    ky = np.fft.fftfreq(Ny, d=dy).reshape(Ny, 1)   # (Ny, 1)

    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid divide-by-zero at DC; we handle DC separately

    # Divergence and scalar curl in Fourier space (using ∂ <-> i2πk)
    div_hat  = 1j * 2*np.pi * (kx * u_hat + ky * v_hat)
    curl_hat = 1j * 2*np.pi * (kx * v_hat - ky * u_hat)

    # Solve Poisson: ΔPhi = div,  ΔPsi = -curl
    # In Fourier: -(2π)^2 k^2 * Phi_hat = div_hat
    Phi_hat = -div_hat / ((2*np.pi)**2 * k2)
    Psi_hat = +curl_hat / ((2*np.pi)**2 * k2)   # because ΔPsi = -curl  ->  -(2π)^2k^2 Psi_hat = -curl_hat

    # Gauge: set DC potential to 0
    Phi_hat[0, 0] = 0.0
    Psi_hat[0, 0] = 0.0

    # Reconstruct ∇Phi and ∇⊥Psi in Fourier space
    grad_u_hat = 1j * 2*np.pi * kx * Phi_hat
    grad_v_hat = 1j * 2*np.pi * ky * Phi_hat

    rot_u_hat  = 1j * 2*np.pi * ky * Psi_hat
    rot_v_hat  = -1j * 2*np.pi * kx * Psi_hat

    # Back to real space
    F_grad = np.stack([np.fft.ifftn(grad_u_hat).real,
                       np.fft.ifftn(grad_v_hat).real], axis=0)

    F_rot  = np.stack([np.fft.ifftn(rot_u_hat).real,
                       np.fft.ifftn(rot_v_hat).real], axis=0)

    # Harmonic/mean mode: constant vector field = spatial mean of F
    mean_u = u.mean()
    mean_v = v.mean()
    H = np.stack([np.full_like(u, mean_u),
                  np.full_like(v, mean_v)], axis=0)

    # Optionally, potentials in real space
    if return_potentials:
        Phi = np.fft.ifftn(Phi_hat).real
        Psi = np.fft.ifftn(Psi_hat).real
        return F_grad, F_rot, H, Phi, Psi

    return F_grad, F_rot, H
