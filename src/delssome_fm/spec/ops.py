"""The array backend that compiled drift functions run against.

compile.py's walk B emits programs over this interface, so the same compiled model runs on
JAX (simulation, float32, under jit/vmap) or NumPy (GPU-free tests and a cross-check,
float64). A PyTorch backend would be one more class with these methods.

There is one method per DAG primitive, plus the few array operations the coupling channels
and the state layout need, plus `wong_wang`: a fused, numerically stable evaluation of the
Wong-Wang subtree u * Inv(1 - exp(-u)). In float32, 1 - exp(-u) is exactly 0 for
|u| < ~6e-8, so the literal subtree returns inf; the fused kernel computes
u / (-expm1(-u)) and fills the removable singularity at u = 0 with its limit, 1. Nothing
here knows about models.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

Array = Any


class Ops(Protocol):
    def const(self, c: float) -> Array: ...
    def add(self, a: Array, b: Array) -> Array: ...
    def mul(self, a: Array, b: Array) -> Array: ...
    def neg(self, a: Array) -> Array: ...
    def inv(self, a: Array) -> Array: ...
    def exp(self, a: Array) -> Array: ...
    def log(self, a: Array) -> Array: ...
    def sq(self, a: Array) -> Array: ...
    def wong_wang(self, u: Array) -> Array:
        """u / (1 - exp(-u)), finite for every finite u (equals 1 at u = 0)."""
    def column(self, x: Array, i: int) -> Array:
        """x[..., i] of a (..., n) array."""
    def stack(self, xs: Sequence[Array]) -> Array:
        """Stack equal-shape arrays along a new last axis."""
    def matvec(self, m: Array, v: Array) -> Array:
        """(N, N) @ (N,) -> (N,)."""
    def rowsum(self, m: Array) -> Array:
        """(N, N) -> (N,), sum over the last axis."""
    def broadcast(self, a: Array, like: Array) -> Array:
        """`a` broadcast to the shape of `like` (turns a scalar into a per-region array)."""


class NumpyOps:
    """float64 NumPy backend, for tests and cross-checks."""

    dtype = np.float64

    def const(self, c: float) -> Array:
        return np.float64(c)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def neg(self, a):
        return -a

    def inv(self, a):
        return 1.0 / a

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)

    def sq(self, a):
        return a * a

    def wong_wang(self, u):
        u = np.asarray(u, dtype=self.dtype)
        zero = u == 0
        safe = np.where(zero, 1.0, u)
        with np.errstate(over="ignore"):
            return np.where(zero, 1.0, safe / -np.expm1(-safe))

    def column(self, x, i):
        return x[..., i]

    def stack(self, xs):
        return np.stack(xs, axis=-1)

    def matvec(self, m, v):
        return m @ v

    def rowsum(self, m):
        return m.sum(axis=-1)

    def broadcast(self, a, like):
        return np.broadcast_to(a, np.shape(like))


class JaxOps:
    """JAX backend, float32 by default (generation.md §8). Safe under jit and vmap."""

    def __init__(self, dtype: Any = None) -> None:
        import jax.numpy as jnp

        self._jnp = jnp
        self.dtype = jnp.float32 if dtype is None else dtype

    def const(self, c: float) -> Array:
        return self._jnp.asarray(c, dtype=self.dtype)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def neg(self, a):
        return -a

    def inv(self, a):
        return 1.0 / a

    def exp(self, a):
        return self._jnp.exp(a)

    def log(self, a):
        return self._jnp.log(a)

    def sq(self, a):
        return a * a

    def wong_wang(self, u):
        jnp = self._jnp
        zero = u == 0
        safe = jnp.where(zero, 1.0, u)
        return jnp.where(zero, 1.0, safe / -jnp.expm1(-safe))

    def column(self, x, i):
        return x[..., i]

    def stack(self, xs):
        return self._jnp.stack(xs, axis=-1)

    def matvec(self, m, v):
        return m @ v

    def rowsum(self, m):
        return m.sum(axis=-1)

    def broadcast(self, a, like):
        return self._jnp.broadcast_to(a, self._jnp.shape(like))
