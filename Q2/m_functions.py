import numpy as np

# matlab functions -> python equivalent

def GS(uinit, b, Niter, lam, dy, bc_func=None):
    u = uinit   # operate in-place (like MATLAB's overwrite behaviour)
    Nx, Ny = u.shape
    denom = 2.0 * (1.0 + lam)
    dy2 = dy * dy
    b_inn = b[1:-1, 1:-1]

    # Use cached masks if available
    red, black = _get_rb_masks(Nx, Ny)

    for _ in range(Niter):
        for mask in (red, black):
            rhs = (
                lam * (u[2:, 1:-1] + u[:-2, 1:-1])
                + (u[1:-1, 2:] + u[1:-1, :-2])
                - b_inn * dy2
            )
            u[1:-1, 1:-1][mask] = (rhs / denom)[mask]
        if bc_func is not None:
            bc_func(u)

    return u


# Mask cache
_mask_cache = {}
def _get_rb_masks(Nx, Ny):
    key = (Nx, Ny)
    if key not in _mask_cache:
        ii, jj = np.ogrid[1:Nx-1, 1:Ny-1]
        _mask_cache[key] = (((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1))
    return _mask_cache[key]


def residual(u, f, lam, dy):
    Nx, Ny = u.shape
    dy2 = dy * dy
    denom = 2.0 * (1.0 + lam)

    res = np.zeros_like(u)
    res[1:-1, 1:-1] = (
        f[1:-1, 1:-1]
        + u[1:-1, 1:-1] * denom / dy2
        - (u[1:-1, 2:] + u[1:-1, :-2]) / dy2
        - lam * (u[2:, 1:-1] + u[:-2, 1:-1]) / dy2
    )
    return res

def restrict(fine):
    """
    Full-weighting restriction  (fine → coarse).
    Fine grid (Nxf, Nyf) → Coarse grid ((Nxf-1)//2+1, (Nyf-1)//2+1).
    """
    Nxf, Nyf = fine.shape
    n = Nxf - 1   # = rows-1 in MATLAB
    m = Nyf - 1   # = cols-1 in MATLAB
    n2 = n // 2
    m2 = m // 2

    coarse = np.zeros((n2 + 1, m2 + 1))

    # Interior: full-weighting stencil  (direct port of restrict.m)
    coarse[1:n2, 1:m2] = (
          (1./4)  * fine[2:n-1:2,   2:m-1:2]       # centre
        + (1./8)  * fine[1:n-2:2,   2:m-1:2]       # i-1
        + (1./8)  * fine[3:n:2,     2:m-1:2]       # i+1
        + (1./8)  * fine[2:n-1:2,   1:m-2:2]       # j-1
        + (1./8)  * fine[2:n-1:2,   3:m:2]         # j+1
        + (1./16) * fine[1:n-2:2,   1:m-2:2]       # (i-1,j-1)
        + (1./16) * fine[1:n-2:2,   3:m:2]         # (i-1,j+1)
        + (1./16) * fine[3:n:2,     1:m-2:2]       # (i+1,j-1)
        + (1./16) * fine[3:n:2,     3:m:2]         # (i+1,j+1)
    )

    # Boundaries: injection
    coarse[:, 0]  = fine[::2, 0]
    coarse[:, -1] = fine[::2, -1]
    coarse[0, :]  = fine[0, ::2]
    coarse[-1, :] = fine[-1, ::2]

    return coarse


def interpolate(coarse):
    Nxc, Nyc = coarse.shape
    n = Nxc - 1
    m = Nyc - 1
    n2 = 2 * n
    m2 = 2 * m

    fine = np.zeros((n2 + 1, m2 + 1))

    fine[2:n2-1:2, 2:m2-1:2] = coarse[1:n, 1:m]

    fine[2:n2-1:2, 1:m2:2] = (coarse[1:n, 0:m] + coarse[1:n, 1:m+1]) / 2.0

    fine[1:n2:2, 2:m2-1:2] = (coarse[0:n, 1:m] + coarse[1:n+1, 1:m]) / 2.0

    fine[1:n2:2, 1:m2:2] = (
        coarse[0:n, 0:m] + coarse[1:n+1, 1:m+1]
        + coarse[1:n+1, 0:m] + coarse[0:n, 1:m+1]
    ) / 4.0

    return fine


def MultigridV(Uin, f, lam, dy, bc_func=None,
               nu_pre=10, nu_post=10, nu_coarse=10,
               bc_func_coarse=None):
    Nx, Ny = Uin.shape

    Usmooth = GS(Uin, f, nu_pre, lam, dy, bc_func=bc_func)

    res = residual(Usmooth, f, lam, dy)

    reshalf = restrict(res)

    dy_c = 2.0 * dy
    dx_f = dy / np.sqrt(lam / (1.0 + 1e-30))  if abs(lam) > 1e-30 else dy
    dx_c = 2.0 * dx_f
    lam_c = lam * (dy_c / dx_c)**2 / (dy / dx_f)**2  if abs(lam) > 1e-30 else 0.0
    
    lam_c = lam 

    err = GS(np.zeros_like(reshalf), reshalf, nu_coarse, lam_c, dy_c,
             bc_func=bc_func_coarse)

    Usmooth = Usmooth + interpolate(err)
    if bc_func is not None:
        bc_func(Usmooth)

    Uout = GS(Usmooth, f, nu_post, lam, dy, bc_func=bc_func)

    return Uout
