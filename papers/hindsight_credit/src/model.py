"""Tabular "model" for the HCA reproduction.

The paper describes no neural networks: every learned object is a table (paper §5, App. E; official code
``hca_classes.py`` keeps ``pi``, ``V`` and ``h`` as arrays). So this module is thin on purpose -- it holds the
parameter tables and the softmax heads; flax is not needed.

Params (all zero-initialised = uniform policy / uniform hindsight, A-INIT):
  theta[n_obs, A]          policy logits, pi(.|o) = softmax(theta[o])
  psi[n_obs, n_obs, A]     state-hindsight logits, h(.|o_x, o_y) = softmax(psi[o_x, o_y])
  phi[n_obs, n_bins, A]    return-hindsight logits, h_z(.|o_x, bin) = softmax(phi[o_x, bin])
  V[n_obs]                 value table (PG critic; State-HCA bootstrap; trained-but-unused in Return HCA)
  rhat[n_obs, A]           immediate-reward model (paper-pseudocode State-HCA variant only)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .data import N_ACTIONS


class Params(NamedTuple):
    theta: jnp.ndarray
    psi: jnp.ndarray
    phi: jnp.ndarray
    V: jnp.ndarray
    rhat: jnp.ndarray


def init_params(n_obs: int, n_bins: int, n_actions: int = N_ACTIONS) -> Params:
    z = jnp.zeros
    return Params(z((n_obs, n_actions)), z((n_obs, n_obs, n_actions)), z((n_obs, n_bins, n_actions)),
                  z((n_obs,)), z((n_obs, n_actions)))


def policy(theta: jnp.ndarray) -> jnp.ndarray:
    """[n_obs, A] action probabilities."""
    return jax.nn.softmax(theta, axis=-1)


def hindsight(logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(logits, axis=-1)
