## Privacy Correctness Checklist

* ✔  Laplace mechanism conforms:  Δ₁ = {config value}, ε′ = {config}.
* ✔  Gaussian RDP bound verified empirically (see test_gaussian_sanity.py).
* ✔  Composition rule applied additively.
* ✔  σ_t never increases; σ_t+1 = β σ_t (β < 1).
* ✔  Clip norm C_t computed from DP-protected mean ‖g‖ (post-processing invariant).
* ✔  Failure guards trigger on:
      - sigma>0 ∧ clip_norm≤0
      - exploding grads > 1e4
* ✔  Accountant unit tests pass (pytest -q). 