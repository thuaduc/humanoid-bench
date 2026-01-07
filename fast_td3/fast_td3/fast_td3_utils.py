import os

import numpy as np
import torch
import torch.nn as nn

from tensordict import TensorDict
from humanoid_bench.envs.custom_env import flatten_obs, unflatten_obs, get_obs_shapes

class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        When playground_mode=True, critic_observations are treated as a concatenation of
        regular observations and privileged observations, and only the privileged part is stored
        to save memory.
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act), device=device, dtype=torch.float
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        if asymmetric_obs:
            if self.playground_mode:
                # Only store the privileged part of observations (n_critic_obs - n_obs)
                self.privileged_obs_size = n_critic_obs - n_obs
                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # Store full critic observations
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
        self.ptr = 0

    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # Sample base indices
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards after first done
            # This creates a cumulative product that zeroes out rewards after the first done
            done_masks = torch.cumprod(
                1.0 - all_dones, dim=2
            )  # [n_env, batch_size, n_step]

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations
        return out


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None, center=True):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.center = center
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training:
            self.update(x)
        if self.center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values using Welford's online algorithm"""
        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)

        # Update count
        new_count = self.count + batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean += (batch_size / new_count) * delta

        # Update variance using Welford's parallel algorithm
        if self.count > 0:  # Ensure we're not dividing by zero
            # Compute batch variance
            batch_var = torch.mean((x - batch_mean) ** 2, dim=0, keepdim=True)

            # Combine variances using parallel algorithm
            delta2 = batch_mean - self._mean
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
        else:
            # For first batch, just use batch variance
            self._var = torch.mean((x - self._mean) ** 2, dim=0, keepdim=True)

        self._std = torch.sqrt(self._var)
        self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class ScalarEmpiricalNormalization(nn.Module):
    """Normalize values by scalar standard deviation (with optional centering and per-feature normalization)."""

    def __init__(self, device, eps=1e-2, until=None, center=True):
        """Initialize ScalarEmpiricalNormalization module.

        Args:
            device: Device to place tensors on.
            eps (float): Small value for stability.
            until (int or None): If specified, stop learning after this many samples.
            center (bool): If True, normalize by subtracting mean. Default True.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.center = center
        # Single scalar mean, variance and std for all features
        self.register_buffer("_mean", torch.tensor(0.0, device=device))
        self.register_buffer("_var", torch.tensor(1.0, device=device))
        self.register_buffer("_std", torch.tensor(1.0, device=device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @property
    def mean(self):
        return self._mean.clone()

    @property
    def std(self):
        return self._std.clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by scalar std and optionally by subtracting mean."""
        if self.training:
            self.update(x)
        if self.center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn scalar mean and variance using Welford's online algorithm."""
        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x)
        
        # Update count
        new_count = self.count + batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean += (batch_size / new_count) * delta

        # Update variance using Welford's parallel algorithm
        if self.count > 0:
            # Compute batch variance
            batch_var = torch.mean((x - batch_mean) ** 2)
            
            # m_a and m_b are scalar values
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            delta2 = batch_mean - self._mean
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
        else:
            # For first batch, just use batch variance
            self._var = torch.mean((x - batch_mean) ** 2)

        self._std = torch.sqrt(self._var)
        self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class StructuredEmpiricalNormalization(nn.Module):
    POSITION_KEYS = ["joint_x", "object_x"]
    SKIP_KEYS = []

    def __init__(self, env_name: str, device, eps=1e-2, until=50_000, center=True):
        """Initialize StructuredEmpiricalNormalization module.

        Args:
            env_name (str): Environment name to determine observation structure.
            device: Device to place the normalizers on.
            eps (float): Small value for stability.
            until (int or None): If specified, stop learning after this many samples.
        """
        super().__init__()
        self.env_name = env_name
        self.device = device
        self.eps = eps
        self.until = until
        self.obs_shapes = get_obs_shapes(env_name)

        # Create normalizers for each feature type (except hardcoded skip keys)
        self.normalizers = nn.ModuleDict()
        for key, shape in self.obs_shapes.items():
            if key in self.SKIP_KEYS:
                continue
            elif key in self.POSITION_KEYS:
                self.normalizers[key] = ScalarEmpiricalNormalization(
                    device=device, eps=eps, until=until, center=center
                )
            else:
                self.normalizers[key] = EmpiricalNormalization(
                    shape=shape, device=device, eps=eps, until=until, center=center
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize dict observations.

        Args:
            x (torch.Tensor): Flat observation tensor.

        Returns:
            torch.Tensor: Normalized flat observation tensor.
        """
        
        x_dict = unflatten_obs(x, self.env_name)
        
        for key, value in x_dict.items():
            if key in self.SKIP_KEYS:
                continue
            elif key in self.normalizers:
                x_dict[key] = self.normalizers[key](value)

        return flatten_obs(x_dict, self.env_name)


class SelectiveEmpiricalNormalization(nn.Module):
    """Normalize observations while freezing specific indices for CUDA graph compatibility.
    
    Automatically identifies which observation field indices to freeze based on hard-coded skip keys.
    """
    SKIP_KEYS = ["joint_x"]

    def __init__(self, env_name: str, device, eps=1e-2, until=None, center=True):
        """Initialize SelectiveEmpiricalNormalization module.

        Args:
            env_name (str): Environment name to determine observation structure.
            device: Device to place the normalizer on.
            eps (float): Small value for stability.
            until (int or None): If specified, stop learning after this many samples.
        """
        super().__init__()
        self.env_name = env_name
        self.device = device
        self.eps = eps
        self.until = until
        self.center = center
        
        # Get observation shapes and build freeze indices
        obs_shapes = get_obs_shapes(env_name)
        
        # Calculate total obs size and build freeze_indices
        total_obs_size = 0
        freeze_indices = []
        index_offset = 0
        
        for key, shape in obs_shapes.items():
            key_size = int(np.prod(shape))
            
            if key in self.SKIP_KEYS:
                # Mark these indices as frozen
                freeze_indices.extend(range(index_offset, index_offset + key_size))
            
            total_obs_size += key_size
            index_offset += key_size
        
        # Create single normalizer for all features with freeze mask
        self.normalizer = EmpiricalNormalization(
            shape=(total_obs_size,), device=device, eps=eps, until=until
        )
        
        # Register freeze mask for CUDA graph compatibility
        if freeze_indices:
            freeze_mask = torch.ones(total_obs_size, dtype=torch.bool, device=device)
            freeze_mask[freeze_indices] = False
            self.register_buffer("_freeze_mask", freeze_mask)
        else:
            self.register_buffer("_freeze_mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize flat observations with frozen indices support.

        Args:
            x (torch.Tensor): Flat observation tensor of shape (batch_size, obs_size).

        Returns:
            torch.Tensor: Normalized flat observation tensor with frozen indices unchanged.
        """
        if x.shape[1:] != self.normalizer._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self.normalizer._mean.shape[1:]}), got {x.shape}"
            )
        
        if self.training:
            self._update_with_freeze(x)
        
        if self.center:
            return (x - self.normalizer._mean) / (self.normalizer._std + self.eps)
        else:
            return x / (self.normalizer._std + self.eps)
    
    @torch.jit.unused
    def _update_with_freeze(self, x):
        """Update normalizer with frozen indices support."""
        if self.normalizer.until is not None and self.normalizer.count >= self.normalizer.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)

        # Update count
        new_count = self.normalizer.count + batch_size

        # Update mean
        delta = batch_mean - self.normalizer._mean
        self.normalizer._mean += (batch_size / new_count) * delta

        # Update variance using Welford's parallel algorithm
        if self.normalizer.count > 0:  # Ensure we're not dividing by zero
            # Compute batch variance
            batch_var = torch.mean((x - batch_mean) ** 2, dim=0, keepdim=True)

            # Combine variances using parallel algorithm
            delta2 = batch_mean - self.normalizer._mean
            m_a = self.normalizer._var * self.normalizer.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.normalizer.count * batch_size / new_count)
            self.normalizer._var = M2 / new_count
        else:
            # For first batch, just use batch variance
            self.normalizer._var = torch.mean((x - self.normalizer._mean) ** 2, dim=0, keepdim=True)

        self.normalizer._std = torch.sqrt(self.normalizer._var)
        self.normalizer.count = new_count
        
        # Restore frozen indices to their initial values (mean=0, std=1)
        if self._freeze_mask is not None:
            self.normalizer._mean[0, ~self._freeze_mask] = 0.0
            self.normalizer._var[0, ~self._freeze_mask] = 1.0
            self.normalizer._std[0, ~self._freeze_mask] = 1.0


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    obs_normalizer,
    critic_obs_normalizer,
    args,
    save_path,
):
    """Save model parameters and training configuration to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")


class SimpleReplayBufferGNN(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        env_name=str,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        When playground_mode=True, critic_observations are treated as a concatenation of
        regular observations and privileged observations, and only the privileged part is stored
        to save memory.
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act), device=device, dtype=torch.float
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        if asymmetric_obs:
            if self.playground_mode:
                # Only store the privileged part of observations (n_critic_obs - n_obs)
                self.privileged_obs_size = n_critic_obs - n_obs
                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # Store full critic observations
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
        self.ptr = 0

    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            env_ids = (
                torch.arange(self.n_env, device=self.device)
                .unsqueeze(1)
                .expand(-1, batch_size)
            )
            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # Sample base indices
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            env_ids = (
                torch.arange(self.n_env, device=self.device)
                .unsqueeze(1)
                .expand(-1, batch_size)
            )

            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards after first done
            # This creates a cumulative product that zeroes out rewards after the first done
            done_masks = torch.cumprod(
                1.0 - all_dones, dim=2
            )  # [n_env, batch_size, n_step]

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations
        return out
