# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import git
import importlib
import os
import pathlib
import torch
import warnings
from tensordict import TensorDict
from typing import Callable
import numpy as np
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip((input - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())
            

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolves the activation function from the name.

    Args:
        act_name: The name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolves the optimizer from the name.

    Args:
        optimizer_name: The name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def split_and_pad_trajectories(
    tensor: torch.Tensor | TensorDict, dones: torch.Tensor
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length of the longest
    trajectory. Returns masks corresponding to valid parts of the trajectories.

    Example:
        Input: [[a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]]

        Output:[[a1, a2, a3, a4], | [[True, True, True, True],
                [a5, a6, 0, 0],   |  [True, True, False, False],
                [b1, b2, 0, 0],   |  [True, True, False, False],
                [b3, b4, b5, 0],  |  [True, True, True, False],
                [b6, 0, 0, 0]]    |  [True, False, False, False]]

    Assumes that the input has the following order of dimensions: [time, number of envs, additional dimensions]
    """

    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # split the tensor into trajectories
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # add at least one full length trajectory
            trajectories = trajectories + (torch.zeros(v.shape[0], *v.shape[2:], device=v.device),)
            # pad the trajectories to the length of the longest trajectory
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)
            # remove the added tensor
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)]
        )
    else:
        # split the tensor into trajectories
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # add at least one full length trajectory
        trajectories = trajectories + (torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device),)
        # pad the trajectories to the length of the longest trajectory
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
        # remove the added tensor
        padded_trajectories = padded_trajectories[:, :-1]
    # create masks for the valid parts of the trajectories
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            t = repo.head.commit.tree
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name: The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.

    Returns:
        The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """Validates the observation configuration and defaults missing observation sets.

    The input is an observation dictionary `obs` containing observation groups and a configuration dictionary
    `obs_groups` where the keys are the observation sets and the values are lists of observation groups.

    The configuration dictionary could for example look like:
        {
            "policy": ["group_1", "group_2"],
            "critic": ["group_1", "group_3"]
        }

    This means that the 'policy' observation set will contain the observations "group_1" and "group_2" and the
    'critic' observation set will contain the observations "group_1" and "group_3". This function will check that all
    the observations in the 'policy' and 'critic' observation sets are present in the observation dictionary from the
    environment.

    Additionally, if one of the `default_sets`, e.g. "critic", is not present in the configuration dictionary,
    this function will:

    1. Check if a group with the same name exists in the observations and assign this group to the observation set.
    2. If 1. fails, it will assign the observations from the 'policy' observation set to the default observation set.

    Args:
        obs: Observations from the environment in the form of a dictionary.
        obs_groups: Observation sets configuration.
        default_sets: Reserved observation set names used by the algorithm (besides 'policy').
            If not provided in 'obs_groups', a default behavior gets triggered.

    Returns:
        The resolved observation groups.

    Raises:
        ValueError: If any observation set is an empty list.
        ValueError: If any observation set contains an observation term that is not present in the observations.
    """
    # check if policy observation set exists
    if "policy" not in obs_groups.keys():
        if "policy" in obs:
            obs_groups["policy"] = ["policy"]
            warnings.warn(
                "The observation configuration dictionary 'obs_groups' must contain the 'policy' key."
                " As an observation group with the name 'policy' was found, this is assumed to be the observation set."
                " Consider adding the 'policy' key to the 'obs_groups' dictionary for clarity."
                " This behavior will be removed in a future version."
            )
        else:
            raise ValueError(
                "The observation configuration dictionary 'obs_groups' must contain the 'policy' key."
                f" Found keys: {list(obs_groups.keys())}"
            )

    # check all observation sets for valid observation groups
    for set_name, groups in obs_groups.items():
        # check if the list is empty
        if len(groups) == 0:
            msg = f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list."
            if set_name in default_sets:
                if set_name not in obs:
                    msg += " Consider removing the key to default to the observations used for the 'policy' set."
                else:
                    msg += (
                        f" Consider removing the key to default to the observation '{set_name}' from the environment."
                    )
            raise ValueError(msg)
        # check groups exist inside the observations from the environment
        for group in groups:
            if group not in obs:
                raise ValueError(
                    f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                    f" environment. Available observations from the environment: {list(obs.keys())}"
                )

    # fill missing observation sets
    for default_set_name in default_sets:
        if default_set_name not in obs_groups.keys():
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' must contain the '{default_set_name}' key."
                    f" As an observation group with the name '{default_set_name}' was found, this is assumed to be the"
                    f" observation set. Consider adding the '{default_set_name}' key to the 'obs_groups' dictionary for"
                    " clarity. This behavior will be removed in a future version."
                )
            else:
                obs_groups[default_set_name] = obs_groups["policy"].copy()
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' must contain the '{default_set_name}' key."
                    f" As the configuration for '{default_set_name}' is missing, the observations from the 'policy' set"
                    f" are used. Consider adding the '{default_set_name}' key to the 'obs_groups' dictionary for"
                    " clarity. This behavior will be removed in a future version."
                )

    # print the final parsed observation sets
    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups
