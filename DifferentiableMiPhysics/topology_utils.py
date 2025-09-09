from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import torch

class InteractionType:
    FIRST = "FIRST"
    SECOND = "SECOND"
    CHECKERED = "CHECKERED"
    DILATED2 = "DILATED2"
    CLIQUE_2x2 = "CLIQUE_2x2"
    # CUSTOM = "CUSTOM"  # Add if needed

OFFSETS_FIRST = [(+1, 0, 0), (0, +1, 0), (0, 0, +1)]
OFFSETS_SECOND = [(0, +1, +1), (+1, 0, +1), (+1, +1, 0), (+1, +1, +1)]
OFFSETS_CHECKERBOARD_EVEN = [(+1, +1, 0), (+1, 0, +1), (0, +1, +1)]
OFFSETS_DILATED2 = [(+2, 0, 0), (0, +2, 0), (0, 0, +2)]  # “skip” springs

SUPPORTED_INTERACTION_TYPES = [
    InteractionType.FIRST,
    InteractionType.SECOND,
    InteractionType.CHECKERED,
    InteractionType.DILATED2,
    InteractionType.CLIQUE_2x2,
    # InteractionType.CUSTOM,
]

def build_grid_nodes(dimX: int, dimY: int, dimZ: int,
                     dist: float,
                     mass: float = 1.0,
                     radius: float = 20.0,
                     fixed_corners: bool = False,
                     drivers: Optional[torch.Tensor] = None,
                     listeners: Optional[torch.Tensor] = None,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """Create [N,8] node feature tensor with grid positions and flags.
    drivers/listeners are lists of (i,j,k) indices.
    """
    N = dimX * dimY * dimZ
    nodes = torch.zeros(N, 8, device=device)
    def idx(i,j,k):
        return (i * dimY + j) * dimZ + k
    for i in range(dimX):
        for j in range(dimY):
            for k in range(dimZ):
                n = idx(i,j,k)
                nodes[n, 0:3] = torch.tensor([i*dist, j*dist, k*dist], device=device)
                nodes[n, 3] = mass
                nodes[n, 4] = radius
                # Fixed corners if requested
                if fixed_corners and (i in (0, dimX-1)) and (j in (0, dimY-1)) and (k in (0, dimZ-1)):
                    nodes[n, 5] = 1.0
    # Mark drivers/listeners
    if drivers is not None:
        nodes[idx(drivers[:,0], drivers[:,1], drivers[:,2]), 6] = 1.0
    if listeners is not None:
        nodes[idx(listeners[:,0], listeners[:,1], listeners[:,2]), 7] = 1.0
    return nodes


def build_edges_nearest(dimX: int, dimY: int, dimZ: int,
                         dist: float,
                         stiffness: float = 1e-3,
                         damping: float = 0.0,
                         device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build 6-neighborhood springs (+/- X,Y,Z) with rest_length=dist.
    Returns (edge_index[2,E], springs[E,6]).
    """
    def idx(i,j,k):
        return (i * dimY + j) * dimZ + k

    edges = []
    attrs = []
    for i in range(dimX):
        for j in range(dimY):
            for k in range(dimZ):
                n = idx(i,j,k)
                for di,dj,dk in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    i2,j2,k2 = i+di, j+dj, k+dk
                    if 0 <= i2 < dimX and 0 <= j2 < dimY and 0 <= k2 < dimZ:
                        m = idx(i2,j2,k2)
                        # add directed edge n->m (we will accumulate forces symmetrically)
                        edges.append((n, m))
                        attrs.append([stiffness, damping, dist, di, dj, dk])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()  # [2,E]
    springs = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, springs

def build_edges_by_type(dimX, dimY, dimZ, dist, stiffness=1e-3, damping=0.0, interaction_type="FIRST", device=None):
    """
    Build edge_index and springs based on the specified interaction_type.
    Supported types: "FIRST", "SECOND", "CHECKERED", "DILATED2", "CLIQUE_2x2"
    """
    if interaction_type == "FIRST":
        return build_edges_nearest(dimX, dimY, dimZ, dist, stiffness, damping, device)
    elif interaction_type == "SECOND":
        edge_index, springs = build_edges_nearest(dimX, dimY, dimZ, dist, stiffness, damping, device)
        edge_index2, springs2 = build_edges_second_neighbor(dimX, dimY, dimZ, dist, stiffness, damping, device)
        edge_index = torch.cat([edge_index, edge_index2], dim=1)
        springs = torch.cat([springs, springs2], dim=0)
        return edge_index, springs
    elif interaction_type == "CHECKERED":
        # Checkerboard and wireframe
        edge_index, springs = build_edges_checkerboard(dimX, dimY, dimZ, dist, stiffness, damping, device)
        return wrap_wireframe(edge_index, springs, dimX, dimY, dimZ, dist, stiffness, damping, device)
    elif interaction_type == "DILATED2":
        # Dilated and wireframe
        edge_index2, springs2 = build_edges_dilated2(dimX, dimY, dimZ, dist, stiffness, damping, device)
        return wrap_wireframe(edge_index2, springs2, dimX, dimY, dimZ, dist, stiffness, damping, device)
    elif interaction_type == "CLIQUE_2x2":
        edge_index, springs = build_edges_clique_2x2(dimX, dimY, dimZ, dist, stiffness, damping, device)
        return wrap_wireframe(edge_index, springs, dimX, dimY, dimZ, dist, stiffness, damping, device)
    else:
        raise ValueError(f"Unknown interaction_type: {interaction_type}")

# You will need to implement these additional topology functions:
def build_edges_second_neighbor(dimX, dimY, dimZ, dist, stiffness, damping, device):
    # Example: connect all second neighbors (diagonal in 3D grid)
    def idx(i,j,k): return (i * dimY + j) * dimZ + k
    edges, attrs = [], []
    for i in range(dimX):
        for j in range(dimY):
            for k in range(dimZ):
                n = idx(i,j,k)
                for di,dj,dk in [(1,1,0),(1,0,1),(0,1,1),(1,1,1),(-1,-1,0),(-1,0,-1),(0,-1,-1),(-1,-1,-1)]:
                    i2,j2,k2 = i+di, j+dj, k+dk
                    if 0 <= i2 < dimX and 0 <= j2 < dimY and 0 <= k2 < dimZ:
                        m = idx(i2,j2,k2)
                        edges.append((n, m))
                        attrs.append([stiffness, damping, dist * (di**2 + dj**2 + dk**2)**0.5, di, dj, dk])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    springs = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, springs

def build_edges_checkerboard(dimX, dimY, dimZ, dist, stiffness, damping, device):
    # Example: connect checkerboard pattern (even sum of indices)
    def idx(i,j,k): return (i * dimY + j) * dimZ + k
    edges, attrs = [], []
    for i in range(dimX):
        for j in range(dimY):
            for k in range(dimZ):
                if (i + j + k) % 2 == 0:
                    n = idx(i,j,k)
                    for di,dj,dk in [(1,1,0),(1,0,1),(0,1,1)]:
                        i2,j2,k2 = i+di, j+dj, k+dk
                        if 0 <= i2 < dimX and 0 <= j2 < dimY and 0 <= k2 < dimZ:
                            m = idx(i2,j2,k2)
                            edges.append((n, m))
                            attrs.append([stiffness, damping, dist * (di**2 + dj**2 + dk**2)**0.5, di, dj, dk])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    springs = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, springs

def build_edges_dilated2(dimX, dimY, dimZ, dist, stiffness, damping, device):
    # Example: connect every second neighbor along axes
    def idx(i,j,k): return (i * dimY + j) * dimZ + k
    edges, attrs = [], []
    for i in range(dimX):
        for j in range(dimY):
            for k in range(dimZ):
                n = idx(i,j,k)
                for di,dj,dk in [(2,0,0),(0,2,0),(0,0,2),(-2,0,0),(0,-2,0),(0,0,-2)]:
                    i2,j2,k2 = i+di, j+dj, k+dk
                    if 0 <= i2 < dimX and 0 <= j2 < dimY and 0 <= k2 < dimZ:
                        m = idx(i2,j2,k2)
                        edges.append((n, m))
                        attrs.append([stiffness, damping, dist * abs(di+dj+dk), di, dj, dk])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    springs = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, springs

def build_edges_clique_2x2(dimX, dimY, dimZ, dist, stiffness, damping, device):
    # Example: fully connect all nodes in each 2x2x2 block
    def idx(i,j,k): return (i * dimY + j) * dimZ + k
    edges, attrs = [], []
    for cx in range((dimX+1)//2):
        for cy in range((dimY+1)//2):
            for cz in range((dimZ+1)//2):
                block = []
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2 if dimZ > 1 else 1):
                            i, j, k = cx*2+dx, cy*2+dy, cz*2+dz
                            if i < dimX and j < dimY and k < dimZ:
                                block.append((i,j,k))
                # Fully connect all pairs in block
                for a in range(len(block)):
                    n1 = idx(*block[a])
                    for b in range(a+1, len(block)):
                        n2 = idx(*block[b])
                        di = block[b][0] - block[a][0]
                        dj = block[b][1] - block[a][1]
                        dk = block[b][2] - block[a][2]
                        edges.append((n1, n2))
                        attrs.append([stiffness, damping, dist * (di**2 + dj**2 + dk**2)**0.5, di, dj, dk])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    springs = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, springs

def build_edges_wireframe(dimX, dimY, dimZ, dist, stiffness=1e-3, damping=0.0, device=None):
    """
    Connects all boundary nodes to form a wireframe (box) around the grid.
    Returns (edge_index, springs) for the wireframe only.
    """
    def idx(i,j,k): return (i * dimY + j) * dimZ + k
    edges, attrs = [], []
    # X edges (along x, at y/z boundaries)
    for j in [0, dimY-1]:
        for k in [0, dimZ-1]:
            for i in range(dimX-1):
                n1 = idx(i, j, k)
                n2 = idx(i+1, j, k)
                edges.append((n1, n2))
                attrs.append([stiffness, damping, dist, 1, 0, 0])
    # Y edges (along y, at x/z boundaries)
    for i in [0, dimX-1]:
        for k in [0, dimZ-1]:
            for j in range(dimY-1):
                n1 = idx(i, j, k)
                n2 = idx(i, j+1, k)
                edges.append((n1, n2))
                attrs.append([stiffness, damping, dist, 0, 1, 0])
    # Z edges (along z, at x/y boundaries)
    for i in [0, dimX-1]:
        for j in [0, dimY-1]:
            for k in range(dimZ-1):
                n1 = idx(i, j, k)
                n2 = idx(i, j, k+1)
                edges.append((n1, n2))
                attrs.append([stiffness, damping, dist, 0, 0, 1])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    springs = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, springs

# Add wireframe to edges and springs
def wrap_wireframe(edge_index, springs, dimX, dimY, dimZ, dist, stiffness=1e-3, damping=0.0, device=None):
    edge_index_wf, springs_wf = build_edges_wireframe(dimX, dimY, dimZ, dist, stiffness, damping, device)
    edge_index_combined = torch.cat([edge_index, edge_index_wf], dim=1)
    springs_combined = torch.cat([springs, springs_wf], dim=0)
    return edge_index_combined, springs_combined