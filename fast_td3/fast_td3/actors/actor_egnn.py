import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_td3.actors.gnn.egnn import EGNN, EGNN_V3
from fast_td3.robots.graph_builder import env_without_object


class ActorEGNN(nn.Module):
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
        attention: bool = False,
        coords_agg: str = "mean",
        normalize: bool = False,
        tanh: bool = False,
        residual: bool = True,
        coord_norm: bool = False,
    ):
        super().__init__()
        self.n_envs = num_envs

        match act_fn:
            case "leaky_relu":
                act_fn = nn.LeakyReLU()
            case "silu":
                act_fn = nn.SiLU()
            case "relu":
                act_fn = nn.ReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        if env_name in env_without_object:
            object_feature_dim = 13
        elif env_name == "h1-balance_simple-v0" or env_name == "h1-sit_hard-v0":
            object_feature_dim = 26
        elif env_name == "h1-reach-v0" or env_name == "h1-push-v0":
            object_feature_dim = 19
        elif env_name == "h1-door-v0":
            object_feature_dim = 17
        else:
            raise ValueError(f"Unsupported environment name: {env_name}")

        # EGNN for message passing
        self.egnn = EGNN_V3(
            hidden_nf=hidden_dim,
            in_joint_nf=2,
            object_feature_dim=object_feature_dim,
            out_node_nf=1,
            batch_size=batch_size,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            robot=robot,
            attention=attention,
            coords_agg=coords_agg,
            normalize=normalize,
            tanh=tanh,
            env_name=env_name,
            residual=residual,
            coord_norm=coord_norm,
        )
        
        # self.egnn = torch.compile(self.egnn, dynamic=True, mode="max-autotune", fullgraph=True)

        # Initialize noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))

    def forward(self, obs, xanchor) -> torch.Tensor:
        result = self.egnn(obs, xanchor)

        return result

    def explore(
        self, obs: torch.Tensor, xanchor: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        act = self(obs, xanchor)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise

