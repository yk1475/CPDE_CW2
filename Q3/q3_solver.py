import numpy as np
from time import perf_counter

GAMMA = 1.4


def dYb_dx(x):
    """Derivative of the bump Yb(x) = 1 - x^2  =>  dYb/dx = -2x."""
    g = np.zeros_like(x)
    mask = (x >= -1.0) & (x <= 1.0)
    g[mask] = -2.0 * x[mask]
    return g


def apply_bcs(phi, g_bottom, dx, dy):
    """
    Apply Neumann BCs on all four boundaries using
    second-order one-sided finite differences.

    Bottom (j=0) :  dphi/dy = g_bottom(x)
    Top    (j=-1):  dphi/dy = 0
    Left   (i=0) :  dphi/dx = 0
    Right  (i=-1):  dphi/dx = 0
    """
    # Bottom: (-3 phi_{i,0} + 4 phi_{i,1} - phi_{i,2}) / (2 dy) = g
    phi[:, 0] = (4.0 * phi[:, 1] - phi[:, 2] - 2.0 * dy * g_bottom) / 3.0
    # Top: dphi/dy = 0
    phi[:, -1] = (4.0 * phi[:, -2] - phi[:, -3]) / 3.0
    # Left: dphi/dx = 0
    phi[0, :] = (4.0 * phi[1, :] - phi[2, :]) / 3.0
    # Right: dphi/dx = 0
    phi[-1, :] = (4.0 * phi[-2, :] - phi[-3, :]) / 3.0


def fix_gauge(phi, anchor=(0, 0)):
    """Fix uniqueness by pinning phi at anchor to zero."""
    phi -= phi[anchor[0], anchor[1]]


#  Gauss-Seidel
def gs_sweep_elliptic(phi, K, gamma, dx, dy, Nx, Ny):
    """
    One GS sweep assuming purely elliptic (central differences everywhere).
    Suitable for large K where K - (gamma+1)*phi_x > 0 everywhere.
    """
    lam = (dy / dx) ** 2
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            phi_x = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * dx)
            alpha = K - (gamma + 1.0) * phi_x
            al = alpha * lam
            denom = 2.0 * (al + 1.0)
            if abs(denom) < 1e-14:
                continue
            phi[i, j] = (al * (phi[i + 1, j] + phi[i - 1, j])
                         + phi[i, j + 1] + phi[i, j - 1]) / denom


def gs_sweep_transonic(phi, K, gamma, dx, dy, Nx, Ny):
    """
    One GS sweep with Murman-Cole type switching:
      alpha > 0  => central difference for phi_xx  (elliptic)
      alpha <= 0 => backward difference for phi_xx  (hyperbolic / upwind)
    """
    lam = (dy / dx) ** 2
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            phi_x = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * dx)
            alpha = K - (gamma + 1.0) * phi_x
            al = alpha * lam

            if alpha > 0.0:
                # ── Elliptic: central difference for phi_xx ──
                denom = 2.0 * (al + 1.0)
                if abs(denom) < 1e-14:
                    continue
                phi[i, j] = (al * (phi[i + 1, j] + phi[i - 1, j])
                             + phi[i, j + 1] + phi[i, j - 1]) / denom
            else:
                # ── Hyperbolic: backward difference in x ──
                # phi_xx = (phi_{i-2} - 2 phi_{i-1} + phi_i) / dx^2
                if i >= 2:
                    denom = al - 2.0   # always < -2 since al < 0
                    phi[i, j] = (2.0 * al * phi[i - 1, j]
                                 - al * phi[i - 2, j]
                                 - phi[i, j + 1]
                                 - phi[i, j - 1]) / denom
                else:
                    # Fallback to central at i=1 (boundary-adjacent)
                    denom = 2.0 * (al + 1.0)
                    if abs(denom) < 1e-14:
                        continue
                    phi[i, j] = (al * (phi[i + 1, j] + phi[i - 1, j])
                                 + phi[i, j + 1] + phi[i, j - 1]) / denom


#  Residual computation
def compute_residual(phi, K, dx, dy, gamma=GAMMA, consistent=False):
    """
    L-infinity norm of the PDE residual at interior points.
    """
    Nx, Ny = phi.shape

    phi_x = np.zeros_like(phi)
    phi_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)

    phi_yy = np.zeros_like(phi)
    phi_yy[:, 1:-1] = (phi[:, 2:] - 2.0 * phi[:, 1:-1] + phi[:, :-2]) / dy**2

    alpha = K - (gamma + 1.0) * phi_x

    if not consistent:
        # Pure central-difference phi_xx
        phi_xx = np.zeros_like(phi)
        phi_xx[1:-1, :] = (phi[2:, :] - 2.0 * phi[1:-1, :] + phi[:-2, :]) / dx**2
    else:
        # Murman-Cole consistent: central where alpha>0, backward where alpha<=0
        phi_xx = np.zeros_like(phi)
        # Central everywhere first
        phi_xx[1:-1, :] = (phi[2:, :] - 2.0 * phi[1:-1, :] + phi[:-2, :]) / dx**2
        # Overwrite with backward where alpha <= 0 and i >= 2
        for i in range(2, Nx - 1):
            for j in range(1, Ny - 1):
                if alpha[i, j] <= 0.0:
                    phi_xx[i, j] = (phi[i-2, j] - 2.0*phi[i-1, j] + phi[i, j]) / dx**2

    res = alpha * phi_xx + phi_yy
    return np.max(np.abs(res[2:-1, 1:-1]))


def phi_update_norm(phi_old, phi_new):
    """L-infinity norm of the iterate update."""
    return np.max(np.abs(phi_new - phi_old))


#  Main solver
def solve(K, Nx=161, Ny=81, q=4.0, s=4.0, r=4.0,
          tol=1e-6, max_iter=60000, check_every=200,
          elliptic_only=False, phi_init=None, verbose=True):
    gamma = GAMMA
    x = np.linspace(-q, s, Nx)
    y = np.linspace(0.0, r, Ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    g_bottom = dYb_dx(x)

    # ── Initialise ──
    if phi_init is not None:
        phi = phi_init.copy()
    else:
        phi = np.zeros((Nx, Ny))
    apply_bcs(phi, g_bottom, dx, dy)
    fix_gauge(phi)

    # ── Choose sweep function ──
    sweep = gs_sweep_elliptic if elliptic_only else gs_sweep_transonic
    use_consistent_res = not elliptic_only

    t0 = perf_counter()
    hist = []
    res = np.inf
    upd = np.inf

    for it in range(1, max_iter + 1):
        if it % check_every == 0 or it == 1:
            phi_old = phi.copy()

        sweep(phi, K, gamma, dx, dy, Nx, Ny)
        apply_bcs(phi, g_bottom, dx, dy)
        fix_gauge(phi)

        if it % check_every == 0 or it == 1:
            res = compute_residual(phi, K, dx, dy, gamma,
                                   consistent=use_consistent_res)
            upd = phi_update_norm(phi_old, phi)
            elapsed = perf_counter() - t0
            hist.append({"it": it, "res": res, "upd": upd, "time": elapsed})

            if verbose and (it == 1 or it % (check_every * 10) == 0):
                print(f"  it {it:6d}  |res|={res:.3e}  |upd|={upd:.3e}  t={elapsed:.1f}s")

            # Convergence: require residual < tol, 
            # OR if transonic accept update-only convergence (residual may not reach tol at shocks even with consistent stencil)
            if res < tol and upd < tol:
                if verbose:
                    print(f"  Converged at it={it}, |res|={res:.3e}, time={elapsed:.2f}s")
                break
            if upd < tol * 1e-2 and use_consistent_res:
                if verbose:
                    print(f"  Converged (update) at it={it}, |res|={res:.3e}, "
                          f"|upd|={upd:.3e}, time={elapsed:.2f}s")
                break
    else:
        elapsed = perf_counter() - t0
        if verbose:
            print(f"  NOT converged after {max_iter} iters, |res|={res:.3e}")

    converged = (res < tol) or (upd < tol * 1e-2 and use_consistent_res)
    info = {
        "converged": converged,
        "iters": it,
        "seconds": perf_counter() - t0,
        "hist": hist,
        "K": K,
        "Nx": Nx, "Ny": Ny,
        "dx": dx, "dy": dy,
        "x": x, "y": y,
        "q": q, "s": s, "r": r,
        "tol": tol,
    }
    return phi, info


#  Post-processing helpers

def compute_phi_x(phi, dx):
    """phi_x via central differences (second-order one-sided at boundaries)."""
    px = np.zeros_like(phi)
    px[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
    px[0, :]    = (-3*phi[0,:] + 4*phi[1,:] - phi[2,:])     / (2.0*dx)
    px[-1, :]   = ( 3*phi[-1,:] - 4*phi[-2,:] + phi[-3,:])  / (2.0*dx)
    return px


def compute_phi_y(phi, dy):
    """phi_y via central differences (second-order one-sided at boundaries)."""
    py = np.zeros_like(phi)
    py[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)
    py[:, 0]    = (-3*phi[:,0] + 4*phi[:,1] - phi[:,2])     / (2.0*dy)
    py[:, -1]   = ( 3*phi[:,-1] - 4*phi[:,-2] + phi[:,-3]) / (2.0*dy)
    return py


def compute_psi(phi_x_field, K, gamma=GAMMA):
    """psi = K - (gamma+1)*phi_x  (>0 elliptic, <0 hyperbolic)."""
    return K - (gamma + 1.0) * phi_x_field


def surface_phi_x(phi, dx):
    """Extract phi_x along y=0 (j=0 column)."""
    px = compute_phi_x(phi, dx)
    return px[:, 0]
