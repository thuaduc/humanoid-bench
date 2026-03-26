
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fast_td3.actors.gnn.egnn_v2 import EGNN_V2

class MockRobot:
    def __init__(self):
        self.num_joints = 19
        self.num_edges = 18
        # Simple linear chain for testing
        self.joint_connections = [(i, i+1) for i in range(18)]

def test_cross_edges():
    device = "cpu"
    batch_size = 3
    num_joints = 19
    num_objs = 2
    
    # Mock EGNN
    model = EGNN_V2(
        in_joint_nf=2,
        in_object_nf=10,
        hidden_nf=64,
        out_node_nf=19,
        in_edge_nf=0,
        device=device,
        batch_size=batch_size,
        act_fn=torch.nn.SiLU(),
        n_layers=1,
        robot="h1",
        env_name="h1-push-v0", # Has objects
    )
    
    print(f"Testing with Batch Size: {batch_size}, Num Joints: {num_joints}, Num Objs: {num_objs}")
    
    # Generate cross edges
    # The function signature in the code is: generate_cross_edges(self, batch_size, num_objs, device)
    edges = model.generate_cross_edges(batch_size, num_objs, device)
    
    src, dst = edges
    print(src)
    print(dst)
    
    print(f"Generated {edges.shape[1]} edges.")
    
    # Check edges for the last batch element
    # Last batch index is 3.
    # Its joints are indices: 3*19 to 4*19 - 1  => 57 to 75
    # Its objects are indices: 4*19 + 3*2 to ... => 76 + 6 = 82, 83
    
    # Let's verify where the joints of the last batch are connecting to.
    last_batch_joint_start = (batch_size - 1) * num_joints
    last_batch_joint_end = batch_size * num_joints
    
    # Filter edges where dst is in the last batch
    mask = (dst >= last_batch_joint_start) & (dst < last_batch_joint_end)
    relevant_src = src[mask]
    relevant_dst = dst[mask]
    
    print(f"\nChecking edges for Batch {batch_size-1} (Joints {last_batch_joint_start}-{last_batch_joint_end-1})")
    print(f"Sources connected to these joints: {relevant_src.unique().tolist()}")
    
    # Expected sources: The objects for batch 3.
    # Object start index = batch_size * num_joints = 76
    # Batch 0 objects: 76, 77
    # Batch 1 objects: 78, 79
    # Batch 2 objects: 80, 81
    # Batch 3 objects: 82, 83
    
    expected_objects = []
    for i in range(num_objs):
        expected_objects.append(batch_size * num_joints + (batch_size - 1) * num_objs + i)
    
    print(f"Expected Objects: {expected_objects}")
    
    if relevant_src.unique().tolist() == expected_objects:
        print("SUCCESS: Edges are correctly connected to the corresponding batch objects.")
    else:
        print("FAILURE: Edges are NOT connected to the correct objects.")
        print(f"Actual Objects: {relevant_src.unique().tolist()}")

    # Check for bidirectional edges
    # If bidirectional, we expect edges where src is a joint and dst is an object
    # Filter edges where src is in the last batch
    mask_out = (src >= last_batch_joint_start) & (src < last_batch_joint_end)
    if mask_out.any():
        print("Bidirectional edges found (Joint -> Object).")
        relevant_dst_out = dst[mask_out]
        print(f"Destinations for Joint->Object edges: {relevant_dst_out.unique().tolist()}")
    else:
        print("No bidirectional edges found (Joint -> Object).")

if __name__ == "__main__":
    test_cross_edges()
