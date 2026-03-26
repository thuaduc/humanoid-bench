import torch
import torch.nn as nn

from fast_td3.actors.transformer.transformer import Transformer
from humanoid_bench.envs.custom_env import get_object_feature_dim


class ActorTransformer(nn.Module):
    def __init__(
        self,
        num_envs: int,
        hidden_dim: int,
        batch_size: int,
        device: torch.device,
        n_layers: int,
        act_fn: str,
        env_name: str,
        robot: str = "h1",
        std_min: float = 0.05,
        std_max: float = 0.8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        match act_fn:
            case "leaky_relu":
                act_fn = nn.LeakyReLU()
            case "silu":
                act_fn = nn.SiLU()
            case "relu":
                act_fn = nn.ReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        in_object_nf = get_object_feature_dim(env_name)

        self.transformer = Transformer(
            hidden_nf=hidden_dim,
            in_joint_nf=5,
            in_object_nf=in_object_nf,
            out_node_nf=1,
            batch_size=batch_size,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            robot=robot,
            env_name=env_name,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Initialize noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        result = self.transformer(obs)
        return result

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise
