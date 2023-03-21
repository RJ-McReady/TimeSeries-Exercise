import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
import jax.numpy as jnp, jax
from jax import grad, jit, vmap

def solve_newton(x0, f, df, ddf, eps=1e-3, verbose=True):
    NB_ITER_MAX = 15
    
    x = x0
    h, w = x0.shape
    dfx_v = df(x).reshape(h * w)
    ddfx_m = ddf(x).reshape((h * w, h * w))

    acc = jnp.linalg.solve(ddfx_m, dfx_v)

    λ = jnp.sum(dfx_v * jnp.linalg.solve(ddfx_m, dfx_v))
    print("newton score", λ * λ / 2)

    eps = 1e-3
    # Newton's method
    nb_iter = 0
    while λ * λ / 2 > eps:
        # compute step
        step = - jnp.linalg.solve(ddfx_m, dfx_v).reshape((h, w))

        # line search
        α, β = 0.25, 0.5  # hardcoded, could definitely be improved
        fx = float(f(x))
        α_dfx_step = α * jnp.sum(df(x) * step)
        step_size = 1.
        while f(x + step_size * step) > fx + step_size * α_dfx_step:
            step_size *= β

        # update x
        x += step_size * step

        # update newton score
        dfx_v = df(x).reshape(h * w)
        ddfx_m = ddf(x).reshape((h * w, h * w))
        λ = jnp.dot(dfx_v, jnp.linalg.solve(ddfx_m, dfx_v))
        nb_iter += 1
        if nb_iter > NB_ITER_MAX: break
        if verbose: print("newton score", λ * λ / 2)
    return x

if __name__ == "__main__":
    I = jnp.array(range(9)).reshape((3,3))
    f = lambda x: jnp.linalg.norm(x - I)**2
    f = jit(f)
    df = grad(f)
    ddf = jax.jacfwd(df)
    solve_newton(jnp.zeros((3,3)), f, df, ddf, 1e-5, True)