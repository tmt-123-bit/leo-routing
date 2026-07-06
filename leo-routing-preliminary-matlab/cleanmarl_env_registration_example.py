"""
Example snippet for registering the LEO routing wrapper inside CleanMARL.

This file is NOT meant to be executed as the main trainer.
It is a migration template showing exactly what should be added later into
F:/cleanmarl/cleanmarl/mappo.py (or a shared env factory file) when PyTorch
training is available.

Why keep it here:
- it lives next to our LEO project files on F:/
- it prevents future guesswork when moving from the local wrapper to the real
  CleanMARL trainer
- it documents the exact import and branching logic cleanmarl needs
"""

from __future__ import annotations


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    """Mirror of the environment() helper style used in cleanmarl/mappo.py.

    This is a template only. In the real cleanmarl file, keep the existing pz /
    smaclite / lbf branches, and add the leo branch below.
    """
    if env_type == "leo":
        # In a real integration, this import path can be adjusted in two ways:
        # 1) copy cleanmarl_leo_wrapper.py into the cleanmarl repo (preferred)
        # 2) add the LEO project directory to PYTHONPATH before launching
        from cleanmarl_leo_wrapper import CleanMARLLeoWrapper

        scenario = env_name or "medium_load"
        return CleanMARLLeoWrapper(scenario=scenario)

    # Existing cleanmarl branches stay untouched in the real file, e.g.:
    # elif env_type == "pz":
    #     ...
    # elif env_type == "smaclite":
    #     ...
    # elif env_type == "lbf":
    #     ...
    raise ValueError(f"Unknown env_type: {env_type}")


EXAMPLE_PATCH = r'''
# Inside F:/cleanmarl/cleanmarl/mappo.py

def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(...)
    elif env_type == "smaclite":
        env = SMACliteWrapper(...)
    elif env_type == "lbf":
        env = LBFWrapper(...)
    elif env_type == "leo":
        from cleanmarl_leo_wrapper import CleanMARLLeoWrapper
        env = CleanMARLLeoWrapper(scenario=env_name)
    return env
'''


if __name__ == "__main__":
    print("This file is a reference template, not a standalone trainer.")
    print(EXAMPLE_PATCH)
