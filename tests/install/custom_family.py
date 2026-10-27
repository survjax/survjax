"""Out-of-tree typed family using only public API (spec 4.2/10.3)."""

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import survjax


@survjax.register_family
@dataclass(frozen=True)
class QuadraticSurvival(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        return -self.rate * jnp.asarray(t) ** 2

    def reference_time(self) -> jax.Array:
        return jnp.ones_like(self.rate)


def main() -> None:
    """Emit results for the host test's independent analytical checks."""
    data = survjax.SurvivalData(jnp.array([1.0, 2.0]), jnp.array([True, False]))
    rate = jnp.asarray(0.5)

    def total(r: jax.Array) -> jax.Array:
        return QuadraticSurvival(rate=r).log_likelihood(data).sum()

    print(
        json.dumps(
            {
                "rows": QuadraticSurvival(rate=rate).log_likelihood(data).tolist(),
                "gradient": float(jax.jit(jax.grad(total))(rate)),
                "hessian": float(jax.jit(jax.hessian(total))(rate)),
            }
        )
    )


if __name__ == "__main__":
    main()
