import enum
import networkx as nx
import matplotlib.pyplot as plt
import torch

from fast_td3.robots.h1 import H1
from fast_td3.robots.g1 import G1


@torch.jit.script
def _quat_conjugate_multiply(q_inv: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Specialized function: multiply conjugate of q_inv with q.
    For unit quaternion, conjugate is just negating imaginary parts.
    Faster than computing conjugate then multiplying separately.
    Expects [w, x, y, z] format.
    """
    # q_inv_conj = [w, -x, -y, -z]
    w1, x1, y1, z1 = q_inv[:, 0], -q_inv[:, 1], -q_inv[:, 2], -q_inv[:, 3]
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=1)


# TODO: this currently works for h1, need to generalize for other robots
class GraphBuilder:
    """Utility to build graph tensors and visualize robot topology."""

    def __init__(
        self, env_name, batch_size, device, robot="h1"
    ):
        robot_lower = robot.lower()
        if robot_lower == "h1":
            self.robot = H1()
        elif robot_lower == "g1":
            self.robot = G1()
        else:
            raise NotImplementedError(f"Robot {robot} not implemented.")

        self.device = device
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_edges = self.robot.joint_connections.__len__()
        
        # Pre-allocated identity quaternion [w, x, y, z] = [1, 0, 0, 0]
        # Using expand() for batch handling to avoid CUDA graph breaking
        self._identity_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32
        )

    def generate_input(self, obs: torch.tensor, xanchor: torch.tensor):
        """Generate input with root information as global context."""
        assert obs.shape[1] == 51, f"obs shape: {obs.shape}"
        assert xanchor.shape[1] == 20, f"xanchor shape: {xanchor.shape}"

        # Extract joint features
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 32:].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]

        # Extract root/object features
        h_object = obs[:, 26:32]
        x_object = (xanchor[:, [0]] - xanchor[:, [0]]).reshape(-1, 3)  # [batch, 3]

        return h, x, h_object, x_object

    def generate_input_object(self, obs: torch.tensor, xanchor: torch.tensor):
        """
        Generates graph inputs for Joints (Equivariant) and Root+Objects (Invariant).
        Optimized: uses specialized JIT quaternion ops, pre-allocated identity quaternion.
        """
        assert xanchor.shape[1] == 21, f"xanchor shape: {xanchor.shape}"
        assert obs.shape[1] == 64, f"obs shape: {obs.shape}"
        
        batch_size = obs.shape[0]
        
        # --- 1. Process Joints ---
        x_joints = (xanchor[:, 1:20] - xanchor[:, :1]).reshape(-1, 3)
        h_joints = torch.cat([
            obs[:, 7:26].reshape(-1, 1),   # Joint Pos [batch, 19]
            obs[:, 39:58].reshape(-1, 1),  # Joint Vel [batch, 19]
        ], dim=1)

        # --- 2. Process free base ---
        # Identity quaternion [w, x, y, z] = [1, 0, 0, 0] for root in relative frame
        # Use expand() to broadcast to batch size - this is a view operation, not a copy,
        # which maintains CUDA graph compatibility
        identity_quat = self._identity_quat.to(dtype=obs.dtype).expand(batch_size, -1)
        
        h_root = torch.cat([
            identity_quat,          # [batch, 4] - identity quaternion
            obs[:, 33:39],          # [batch, 6] - root velocities
        ], dim=1)

        # --- 3. Process object ---
        root_quat = obs[:, 3:7]     # [batch, 4]
        obj_quat = obs[:, 29:33]    # [batch, 4]
        
        # Use specialized JIT function: conjugate multiply in one operation
        obj_quat_rel = _quat_conjugate_multiply(root_quat, obj_quat)  # [batch, 4]
        
        h_obj = torch.cat([
            obj_quat_rel,           # [batch, 4] - relative quaternion
            obs[:, 58:64]           # [batch, 6] - object velocities
        ], dim=1)

        # --- 4. Combine Root and Object ---
        # h_objects: [batch, 2, 10] -> [batch*2, 10]
        # Each entity has 10 dims: 4 (quat) + 6 (velocity)
        h_objects = torch.stack([h_root, h_obj], dim=1).reshape(-1, 10)
        x_objects = (xanchor[:, [0, 20]] - xanchor[:, :1]).reshape(-1, 3)

        return h_joints, x_joints, h_objects, x_objects

    def generate_input_for_mixed_type(self, obs: torch.tensor, xanchor: torch.tensor):
        """
        Generates graph inputs for environments with objects using a simpler approach.
        This approach concatenates more observation features for the object node,
        which has been shown to work well for tasks like h1-balance_simple-v0.
        
        Expected dimensions for environments with objects:
        - xanchor: [batch, 21, 3] - root (1) + joints (19) + object (1) = 21 anchor positions
        - obs: [batch, 64] - observation vector for environments with objects
        
        Returns:
            h_node: Joint features [batch*19, 2] - position and velocity per joint
            h_object: Object features [batch, 26] - comprehensive object info
            x_joint: Joint positions relative to root [batch*19, 3]
            x_object: Object position relative to root [batch, 3]
        """
        # Expected: 21 anchor positions (root + 19 joints + 1 object)
        assert xanchor.shape[1] == 21, f"xanchor shape: {xanchor.shape}"
        # Expected: 64-dim observation for environments with objects
        assert obs.shape[1] == 64, f"obs shape: {obs.shape}"

        # Joint positions relative to root
        x_joint = (xanchor[:, 1:20] - xanchor[:, [0]]).reshape(-1, 3)
        # Object position relative to root
        x_object = (xanchor[:, 20:] - xanchor[:, [0]]).reshape(-1, 3)

        # Joint features: position and velocity
        h_node = torch.cat(
            [
                obs[:, 7:26].reshape(-1, 1),   # Joint positions
                obs[:, 39:58].reshape(-1, 1),  # Joint velocities
            ],
            dim=1,
        )

        # Object features: comprehensive info including root and object state
        # obs[:, 0:7] = root position (3) + orientation quaternion (4) = 7
        # obs[:, 26:39] = object position (3) + quaternion (4) + velocity (6) = 13
        # obs[:, 58:64] = additional object features (6)
        h_object = torch.cat([obs[:, 0:7], obs[:, 26:39], obs[:, 58:64]], dim=1)

        return h_node, h_object, x_joint, x_object

    def visualize_graph(
        self, save_path: str = "robot_graph.png"
    ):
        """Visualize the current graph.

        Args:
            save_path: Path to save the generated image.
        """
        G = nx.DiGraph()

        # Determine node ids to add
        num_joint_nodes = len(self.robot.JOINT)
        all_node_ids = list(range(num_joint_nodes))

        # Add nodes with joint/object names as labels
        for nid in all_node_ids:
            G.add_node(nid, label=self.robot.get_joint_name(nid))

        # Add edges
        for edge in self.robot.joint_connections:
            G.add_edge(edge[0], edge[1])

        # Use custom robot-like layout
        pos = self.robot.get_robot_layout_positions()

        # Create labels dictionary
        labels = {nid: self.robot.get_joint_name(nid) for nid in G.nodes()}

        # Define color scheme for different connection types
        connection_colors = self.robot.connection_colors

        # Categorize edges by connection type
        edge_groups = {}
        for edge in G.edges():
            conn_type = self.robot.get_connection_type(edge[0], edge[1])
            if conn_type not in edge_groups:
                edge_groups[conn_type] = []
            edge_groups[conn_type].append(edge)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=400,
            alpha=0.9,
            linewidths=1,
            edgecolors="black",
        )

        # Draw edges by type with different colors and styles
        for conn_type, edges in edge_groups.items():
            color = connection_colors.get(conn_type, "#999999")

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                arrows=True,
                arrowsize=8,
                alpha=0.8,
                width=1,
                style="solid",
                arrowstyle="->",
            )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels, font_size=4, font_weight="bold", font_color="black"
        )

        plt.title(
            f"Humanoid Robot Joint Connection Graph\n(Color-coded by Connection Type)",
            fontsize=10,
            fontweight="bold",
            pad=20,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()


if __name__ == "__main__":
    # Example standalone usage for quick visual checks
    print("Visualizing H1 robot...")
    gb_h1 = GraphBuilder(env_name="h1-run-v0", batch_size=1, device="cpu", robot="h1")
    gb_h1.visualize_graph(save_path="h1_robot_graph.png")
    
    print("Visualizing G1 robot...")
    gb_g1 = GraphBuilder(env_name="g1-run-v0", batch_size=1, device="cpu", robot="g1")
    gb_g1.visualize_graph(save_path="g1_robot_graph.png")
