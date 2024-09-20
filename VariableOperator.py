from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def variable_operator(edge_index: Tensor,
                      w: Tensor,
                      edge_weight: OptTensor = None,
                      dtype: Optional[torch.dtype] = None,
                      num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_weight = edge_weight.to(device)
    w = w.to(device)


    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                     device=edge_index.device)


    num_nodes = maybe_num_nodes(edge_index, num_nodes)


    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    # Compute A_norm =  * (D~^{-1/2} W D~^{-1/2}）。
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_weight = w * edge_weight.clone().detach()


    # Q= I - w*A_norm.
    edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                     fill_value=1., num_nodes=num_nodes)
    assert tmp is not None
    edge_weight = tmp

    return edge_index, edge_weight
