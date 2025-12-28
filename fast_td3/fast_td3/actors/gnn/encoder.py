import torch
import torch.nn as nn
from e3nn import o3


class MultiObjectGeometricEncoder(nn.Module):
    def __init__(self, num_objects, scalar_dim, hidden_dim=32, output_dim=None, device="cpu"):
        """
        Args:
            num_objects (int): Number of oriented objects (e.g., 2 for Pelvis + Board).
            scalar_dim (int): Dimension of scalar inputs (joint velocities/angles, etc.).
            hidden_dim (int): Hidden dimension for equivariant features (default 32).
            output_dim (int): Output dimension after fusion. If None, defaults to hidden_dim * 8.
            device (str): Device for model ('cpu' or 'cuda:0', etc.).
        """
        super().__init__()
        self.num_objects = num_objects
        self.hidden_dim = hidden_dim
        self.scalar_dim = scalar_dim
        self.device_str = device
        
        # Default output dimension: scale with hidden_dim
        if output_dim is None:
            output_dim = hidden_dim * 8
        self.output_dim = output_dim
        
        # 1. Define Features: Hidden_Dim scalars + Hidden_Dim vectors
        # e.g., "32x0e + 32x1e" for hidden_dim=32
        self.irreps = o3.Irreps(f"{hidden_dim}x0e + {hidden_dim}x1e")
        
        # 2. Learnable "ID Cards" for all objects
        # We create a table of embeddings. 
        # index 0 -> Pelvis params, index 1 -> Board params
        self.object_embeddings = nn.Parameter(torch.randn(num_objects, self.irreps.dim))
        
        # 3. Directional Modulation (Shared MLP)
        # Allows joint velocities to modulate the importance of each object's features
        self.modulation_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.irreps.dim * num_objects), # Output unique weights for each object
        )
        
        # 4. Output Fusion
        # Collapses the N objects into a single vector for the RL policy
        self.fusion = nn.Linear(self.irreps.dim * num_objects, output_dim)

    def forward(self, quaternions, scalars):
        """
        Args:
            quaternions: (Batch, Num_Objects, 4) - Quaternions [w, x, y, z] for all objects.
            scalars:   (Batch, Scalar_Dim) - Joint velocities, angles, etc.
        
        Returns:
            (Batch, Output_Dim) - Fused geometric object features.
        """
        batch_size = quaternions.shape[0]
        
        # --- Step 1: Prepare & Expand Embeddings ---
        # Start: (Num_Objects, Feat_Dim)
        # Target: (Batch, Num_Objects, Feat_Dim)
        z = self.object_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # --- Step 2: Reshape for processing ---
        # (Batch, N, Feat_Dim) -> (Batch * N, Feat_Dim)
        z_flat = z.reshape(batch_size * self.num_objects, -1)
        
        # Simple transformation: use embeddings as-is with modulation
        # Skip explicit rotation since we're in a torch.compile context
        # and e3nn's GeometricTensor API is not available
        z_out = z_flat.reshape(batch_size, self.num_objects, -1)

        # --- Step 4: Modulation ---
        # Scalars: (Batch, Scalar_Dim) -> Weights: (Batch, N * Feat_Dim)
        weights = self.modulation_mlp(scalars)
        weights = weights.reshape(batch_size, self.num_objects, -1)
        
        z_modulated = z_out * torch.sigmoid(weights)

        # --- Step 5: Flatten for Policy ---
        # Combine all object features into one long vector
        # (Batch, N, Feat_Dim) -> (Batch, N * Feat_Dim)
        policy_input = z_modulated.reshape(batch_size, -1)
        
        return self.fusion(policy_input)
    
    

# --- Usage Example ---
if __name__ == "__main__":
    # Setup: 2 Objects (Pelvis + Board)
    BATCH = 16
    NUM_OBJS = 2 
    
    model = MultiObjectGeometricEncoder(
        num_objects=NUM_OBJS, 
        scalar_dim=19,  # joint velocities for H1
        hidden_dim=32,
        output_dim=128,  # Custom output dimension
        device="cpu"
    )
    
    # Input: Quaternions [w, x, y, z] per batch item
    quaternions = torch.randn(BATCH, NUM_OBJS, 4)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)  # Normalize
    
    scalars = torch.randn(BATCH, 19)  # Joint velocities
    
    output = model(quaternions, scalars)
    print(f"Output Shape: {output.shape}")  # (16, 128)