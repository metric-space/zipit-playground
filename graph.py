import torch
import networkx as nx
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from graphviz import Digraph

# TODO: figure out what is PREFIX POSTFIX really

class CovarianceMetric():
    name = 'covariance'
    
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
    
    def update(self, batch_size, *feats, **aux_params):
        # jitter the second element in feats
        assert len(feats) == 2
        feats[1] += torch.randn_like(feats[1])
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)
        
        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]
        
        if self.mean  is None: self.mean  = torch.zeros_like( mean)
        if self.outer is None: self.outer = torch.zeros_like(outer)
        if self.std   is None: self.std   = torch.zeros_like(  std)
            
        self.mean  += mean  * batch_size
        self.outer += outer * batch_size
        self.std   += std   * batch_size
    
    def finalize(self, numel, eps=1e-4):
        self.outer /= numel
        self.mean  /= numel
        self.std   /= numel
        cov = self.outer - torch.outer(self.mean, self.mean)
        if torch.isnan(cov).any():
            breakpoint()
        if (torch.diagonal(cov) < 0).sum():
            pdb.set_trace()
        return cov

def match_tensors_zipit(
    metric, r=.5, a=0.3, b=.125, 
    print_merges=False, get_merge_value=False, add_bias=False, 
    **kwargs
):
    """
    ZipIt! matching algorithm. Given metric dict, computes matching as defined in paper. 
    Args:
    - metric: dictionary containing metrics. This must contain either a covariance or cossim matrix, and 
        must be [(num_models x model_feature_dim), (num_models x model_feature_dim)]. 
    - r: Amount to reduce total input feature dimension - this is num_models x model_feature_dim. This function will
        compute (un)merge matrix that goes from 
        (num_models x model_feature_dim) -> (1-r)*(num_models x model_feature_dim) = merged_feature_dim.
        E.g. if num_models=2, model_feature_dim=10 and r=.5, the matrix will map from 2x10=20 -> (1-.5)x2x10=10, or halve the 
        collective feature space of the models.
    - a: alpha hyperparameter as defined in Section 4.3 of our paper. 
    - b: beta hyperparameter as defined in Section 4.3 of our paper.
    - print_merges: whether to print computed (un)merge matrices.
    - get_merge_value default False, returns the sum of correlations over all the merges which the algorithm made. 
    - add_bias: whether to add a bias to the input. This should only be used if your module expects the input with bias offset.
    returns:
    - (un)merge matrices
    """

    def remove_col(x, idx):
        return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

    def compute_correlation(covariance, eps=1e-7):
        std = torch.diagonal(covariance).sqrt()
        covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
        return covariance

    def add_bias_to_mats(mats):
        """ Maybe add bias to input. """
        pad_value = 0
        pad_func = torch.nn.ConstantPad1d((0, 1, 0, 1), pad_value)
        biased_mats = []
        for mat in mats:
            padded_mat = pad_func(mat)
            padded_mat[-1, -1] = 1
            biased_mats.append(padded_mat)
        return biased_mats


    if "covariance" in metric:
        sims = compute_correlation(metric["covariance"])
    elif "cossim" in metric:
        sims = metric["cossim"]
    O = sims.shape[0]
    remainder = int(O * (1-r) + 1e-4)
    permutation_matrix = torch.eye(O, O)#, device=sims.device)

    torch.diagonal(sims)[:] = -torch.inf

    num_models = int(1/(1 - r) + 0.5)
    Om = O // num_models

    original_model = torch.zeros(O, device=sims.device).long()
    for i in range(num_models):
        original_model[i*Om:(i+1)*Om] = i

    to_remove = permutation_matrix.shape[1] - remainder
    budget = torch.zeros(num_models, device=sims.device).long() + int((to_remove // num_models) * b + 1e-4)

    merge_value = []

    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        
        merge_value.append(permutation_matrix[row_idx, col_idx])

        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx
        
        row_origin = original_model[row_idx]
        col_origin = original_model[col_idx]
        
        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = remove_col(permutation_matrix, col_idx)
        
        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])
        
        if 'magnitudes' in metric:
            metric['magnitudes'][row_idx] = torch.minimum(metric['magnitudes'][row_idx], metric['magnitudes'][col_idx])
            metric['magnitudes'] = remove_col(metric['magnitudes'][None], col_idx)[0]
        
        if a <= 0:
            sims[row_origin*Om:(row_origin+1)*Om, row_idx] = -torch.inf
            sims[col_origin*Om:(col_origin+1)*Om, row_idx] = -torch.inf
        else: sims[:, row_idx] *= a
        sims = remove_col(sims, col_idx)
        
        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        if a <= 0:
            sims[row_idx, row_origin*Om:(row_origin+1)*Om] = -torch.inf
            sims[row_idx, col_origin*Om:(col_origin+1)*Om] = -torch.inf
        else: sims[row_idx, :] *= a
        sims = remove_col(sims.T, col_idx).T

        row_origin, col_origin = original_model[row_idx], original_model[col_idx]
        original_model = remove_col(original_model[None, :], col_idx)[0]
        
        if row_origin == col_origin:
            origin = original_model[row_idx].item()
            budget[origin] -= 1

            if budget[origin] <= 0:
                # kill origin
                selector = original_model == origin
                sims[selector[:, None] & selector[None, :]] = -torch.inf
    
    if add_bias:
        unmerge_mats = permutation_matrix.chunk(num_models, dim=0)
        unmerge_mats = add_bias_to_mats(unmerge_mats)
        unmerge = torch.cat(unmerge_mats, dim=0)
    else:
        unmerge = permutation_matrix

    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
    if print_merges:
        O, half_O = unmerge.shape
        A_merge, B_merge = unmerge.chunk(2, dim=0)
        
        A_sums = A_merge.sum(0)
        B_sums = B_merge.sum(0)
        
        A_only = (B_sums == 0).sum()
        B_only = (A_sums == 0).sum()
        
        overlaps = half_O - (A_only + B_only)
        
        print(f'A into A: {A_only} | B into B: {B_only} | A into B: {overlaps}')
        print(f'Average Connections: {unmerge.sum(0).mean()}')
    
    merge = merge.to(sims.device)
    unmerge = unmerge.to(sims.device)
    if get_merge_value:
        merge_value = sum(merge_value) / len(merge_value)
        return merge.T, unmerge, merge_value
    return merge.T, unmerge

class FeatureReshapeHandler:
    """ Instructions to reshape layer intermediates for alignment metric computation. """
    def handle_conv2d(self, x):
        # reshapes conv2d representation from [B, C, H, W] to [C, -1]
        B, C, H, W = x.shape
        return x.permute(1, 0, 2, 3).reshape(C, -1)

    def handle_linear(self, x):
        # x is shape [..., C]. Want [C, -1]
        x = x.flatten(0, len(x.shape)-2).transpose(1, 0).contiguous()
        return x
    
    def __init__(self, class_name, info):
        self.handler = {
            'BatchNorm2d': self.handle_conv2d,
            'LayerNorm': self.handle_linear,
            'Conv2d': self.handle_conv2d,
            'Linear': self.handle_linear,
            'GELU': self.handle_linear,
            'AdaptiveAvgPool2d': self.handle_conv2d,
            'LeakyReLU': self.handle_conv2d,
            'ReLU': self.handle_conv2d, 
            'Tanh': self.handle_conv2d,
            'MaxPool2d': self.handle_conv2d,
            'AvgPool2d': self.handle_conv2d,
            'SpaceInterceptor': self.handle_conv2d,
            'Identity': self.handle_linear,
            
        }[class_name]
        self.info = info


class NodeType(Enum):
    MODULE = 0          # node is torch module
    PREFIX = 1          # node is a PREFIX (i.e., we want to hook inputs to child node)
    POSTFIX = 2         # node is a POSTFIX  (i.e., we want to hook outputs to parent node)
    SUM = 3             # node is a SUM (e.g., point where residual connections are connected - added)
    CONCAT = 4          # node is a CONCATENATION (e.g, point where residual connections are concatenated)
    INPUT = 5           # node is an INPUT (graph starting point)
    OUTPUT = 6          # node is an OUTPUT (graph output point)
    EMBEDDING = 7       # node is an embedding module (these can only be merged)


class Node:
    def __init__(self, name, type, layer_name=None, param_name=None, chunk=None, special_merge=None):
        self.name = name
        self.type = type
        self.layer_name = layer_name
        self.param_name = param_name
        self.chunk = chunk
        self.special_merge = special_merge

    def __str__(self):
        return f"Node({self.name}, {self.type}, {self.layer_name}, {self.param_name}, {self.chunk}, {self.special_merge})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


# directed graph
class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_node(self, node:Node):
        self.nodes.append(node)
        self.edges[node] = []

    def add_edge(self, node1, node2):
        self.edges[node1].append(node2)

    def successors(self, node):
        return self.edges[node]

    def predecessors(self, node):
        preds = []
        for n in self.nodes:
            if node in self.edges[n]:
                preds.append(n)
        return preds
    
    def len(self):
        return len(self.nodes)

    def draw(self):
        dot = Digraph(comment="Model Architecture")

        for node in self.nodes:
            name = node.name
            if node.type == NodeType.MODULE:
                dot.node(node.name, node.layer_name)
            elif node.type == NodeType.PREFIX:
                dot.node(node.name, "Prefix")
            elif node.type == NodeType.POSTFIX:
                dot.node(node.name, "Postfix")
            elif node.type == NodeType.SUM:
                dot.node(node.name, "Sum")
            else:
                dot.node(node.name, node.name)
        for n in self.edges:
            for succ in self.edges[n]:
                dot.edge(n.name, succ.name)
        return dot



class ModelGraph(ABC):

    def __init__(self, model):
        self.reset_graph()
        self.model = model
        self.named_modules = dict(model.named_modules())
        self.named_parameters = dict(model.named_parameters())

        self.hooks = []
        self.intermediates = {} # layer to activations

        self.merged = {}
        self.unmerged = {}

    def reset_graph(self):
        self.G = Graph()

    def preds(self, node):
        return self.G.predecessors(node)

    def succs(self, node):
        return self.G.successors(node)

    def create_node_name(self):
        return str(self.G.len())

    def create_node(self, 
                    node_name=None, 
                    layer_name=None, 
                    param_name=None, 
                    type=NodeType.MODULE,
                    chunk=None,
                    special_merge=None):

        if node_name is None:
            node_name = self.create_node_name()

        node = Node(node_name, type, layer_name, param_name, chunk, special_merge)

        self.G.add_node(node)

        return node

    def create_edge(self, node1, node2):
        self.G.add_edge(node1, node2)

    def get_module(self, name):
        return self.named_modules[name]

    def add_nodes_from_sequence(self, name_prefix, list_of_names, input_node, sep="."):
        source = input_node
        for name in list_of_names:
            if isinstance(name, str):
                temp_node = self.create_node(layer_name=name_prefix + sep + name)
            else:
                temp_node = self.create_node(type=name)
            self.create_edge(source, temp_node)
            source = temp_node
        return source

    def print_prefix(self):
        for node in self.G.nodes:
            if node.type in [NodeType.PREFIX, NodeType.POSTFIX]:
                print(f"{node.name} in={len(self.preds(node))} out={len(self.succs(node))}")

    def hook_fn(self, name):
        def hook(module, input, _):
            a = FeatureReshapeHandler(module.__class__.__name__, module)
            b = a.handler(input[0])
            self.intermediates[name] = b
            print(f"Hooked {name} with shape {b.shape}")
            return None
        return hook

    def add_hooks(self):
        self.clear_hooks()

        for node in self.G.nodes:
            if node.type == NodeType.PREFIX:

                for succ in self.succs(node):
                    print(f"Trying {succ.layer_name} with type {succ.type}")

                    if succ.type == NodeType.MODULE:
                        # add it to the first module
                        self.hooks.append(self.named_modules[succ.layer_name].register_forward_hook(self.hook_fn(succ.layer_name)))
                        break
                else:
                    raise RuntimeError(f"Node type {node.type} not supported for prefix hooks")
            elif node.type == NodeType.POSTFIX:
                for pred in self.preds(node):
                    if pred.type == NodeType.MODULE:
                        self.hooks.append(self.named_modules[pred.layer_name].register_forward_hook(self.hook_fn(pred.layer_name)))
                        break
                raise RuntimeError(f"Node type {node.type} not supported for postfix hooks")


    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []



    def clear_hooks(self):
        """ Clear graph hooks. """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    
    def compute_intermediates(self, x):
        """ Computes all intermediates in a graph network. Takes in a torch tensor (e.g., a batch). """
        self.model = self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.intermediates = {}
            self.model(x)
            return self.intermediates
    
    
    @abstractmethod
    def graphify(self):
        """ 
        Abstract method. This function is implemented by your architecture graph file, and is what actually
        creates the graph for your model. 
        """
        return NotImplemented


class Resnet(ModelGraph):

    def __init__(self, model, shortcut_name='downsample', layer_name='layer', head_name='linear', num_layers=3):
        super().__init__(model)
        self.shortcut_name = shortcut_name
        self.layer_name = layer_name
        self.head_name = head_name
        self.num_layers = num_layers

    def add_basic_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + self.shortcut_name

        shotcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(self.get_module(shortcut_prefix)) > 0:
            input_node = self.add_nodes_from_sequence('', [NodeTypes.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0', '1'],
                input_node=input_node
            )

        skip_node = self.add_nodes_from_sequence(
                name_prefix, 
                ['conv1', 'bn1',NodeType.PREFIX, 'conv2', 'bn2', NodeType.SUM], 
                input_node)

        # join residual connection
        self.create_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_bottleneck_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + self.shortcut_name

        shortcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(self.get_module(shortcut_prefix)) > 0:
            input_node = self.add_nodes_from_sequence('', [NodeTypes.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0', '1'],
                input_node=input_node
            )

        skip_node = self.add_nodes_from_sequence(
                name_prefix, 
                ['conv1', 'bn1', NodeType.PREFIX, 
                 'conv2', 'bn2', NodeType.PREFIX,
                 'conv3', 'bn3', NodeType.SUM], 
                input_node)

        # join residual connection
        self.create_edge(shortcut_output_node, skip_node)

        return skip_node
        
    def add_layer_nodes(self, name_prefix, input_node):
        print(f"Adding layer nodes for {name_prefix}")
        source = input_node

        for layer_index, block in enumerate(self.get_module(name_prefix)):
            block_class = block.__class__.__name__
            print(f"Block class: {block_class}")

            if block_class == 'BasicBlock':
                source = self.add_basic_block_nodes(name_prefix +'.'+ str(layer_index), source)
            elif block_class == 'Bottleneck':
                source = self.add_bottleneck_block_nodes(name_prefix + '.' + str(layer_index), source)
            else:
                raise RuntimeError(f"Block class {block_class} not supported")

        return source

    def graphify(self):
        input_node = self.create_node(type=NodeType.INPUT)
        print(input_node)

        input_node = self.add_nodes_from_sequence('', ['conv1', 'bn1'], input_node, sep='')

        for i in range(1, self.num_layers + 1):
            input_node = self.add_layer_nodes(self.layer_name + str(i), input_node)

        input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX, 'avgpool', self.head_name, NodeType.OUTPUT], input_node, sep='')


        return self

def resnet50(model):
    return Resnet(model, shortcut_name='downsample', head_name='fc', num_layers=4)

if __name__ == '__main__':
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models.resnet as resnet
    # import models.resnets as resnet2

    data_x = torch.rand(4, 3, 224, 224)
    data_y = torch.zeros(4)

    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)

    model = resnet.resnet50()

    graph = resnet50(model).graphify()

    graph.add_hooks()

    intermediates = graph.compute_intermediates(data_x)
    #intermediates = {k: [v,v] for k,v in intermediates.items()}

    metric = CovarianceMetric()

    for k, v in intermediates.items():
        metric.update(1, [v, v])
        print(f"{k}: {metric.finalize(1).shape}")






    #diagram = graph.G.draw()
    #diagram.render('resnet50', format='png')
    



