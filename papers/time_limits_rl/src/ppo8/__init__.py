"""Vendored copies of the differential-tested JAX PPO building blocks (code identical to the source; comments that
cited notes not shipped here were trimmed).

Source: our reproduction of "The 37 Implementation Details of PPO" (papers/ppo_implementation_details in this
repository; original commit 19d3c41bf6756f50c88101f9bc8915c0e085446d), files src/algorithm.py, src/model.py,
src/data.py. Those files were differential-tested against CleanRL @fe8d8a0, the 37-details post's code
and openai/baselines @ea25b9e (ppo2 GAE, clipped losses, Adam/TF1 epsilon, VecNormalize). Their code
is not edited here (AST-identical to the source commit): tests/test_vendor.py pins their SHA-256 so any drift fails
loudly. Only the pure functions are used (GAE reference, losses, optimizer,
networks, distributions, running-mean-std and baselines reward scaling); the gymnax/Brax VecEnv of data.py is not.
"""
VENDOR_COMMIT = "19d3c41bf6756f50c88101f9bc8915c0e085446d"
VENDOR_SHA256 = {
    "algorithm.py": "6b1e5529a62390153c3f6ac10d935b4123ef6126ad4b5e848c19eb2808625bb2",
    "model.py": "fd97d4aa815ee23aa9f43673675dbbea7b531a0668b0d06a94c410b8f196d207",
    "data.py": "49a986622bad57de5eec80ab5d2efd7906dec9c1f9f87368fa99906910a3ee91",
}
