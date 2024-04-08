import torch
import networkx as nx
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from graphviz import Digraph
import click
from tqdm import tqdm


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

    def reshape(self, x):
        x = self.handler(x)

        # Handle modules that we only want a piece of
        if self.info['chunk'] is not None:
            idx, num_chunks = self.info['chunk']
            x = x.chunk(num_chunks, dim=0)[idx]

        return x


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
        self.color = None
        self.traversal_number = None

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

    def decolor(self):
        for node in self.nodes:
            node.color = None

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

            color = {}
            if node.type == NodeType.MODULE:
                t = (node.name, node.layer_name + str(node.traversal_number) or "")
            elif node.type == NodeType.PREFIX:
                t = (node.name, "Prefix" + str(node.traversal_number) or "")
            elif node.type == NodeType.POSTFIX:
                t = (node.name, "Postfix" + str(node.traversal_number) or "")
            elif node.type == NodeType.SUM:
                t = (node.name, "Sum" + str(node.traversal_number) or "")
            else:
                t = (node.name, node.name)
                if node.name and node.traversal_number:
                    t = (node.name, node.name + str(node.traversal_number) or "")
            if node.color is not None:
                color = {'color': node.color, 'style': 'filled'}
            #print(f"Adding node {t} with color {color}")
            dot.node(*t, **color)
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

        self.merged = set()
        self.unmerged = set()

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

    def add_edge(self, node1, node2):
        self.G.add_edge(node1, node2)

    def get_module(self, name):
        return self.named_modules[name]

    def add_nodes_from_sequence(self, name_prefix, list_of_names, input_node, sep="."):
        source = input_node
        for name in list_of_names:
            if isinstance(name, str):
                if name_prefix == '':
                    temp_node = self.create_node(layer_name=name)
                else:
                    temp_node = self.create_node(layer_name=name_prefix + sep + name)
            else:
                temp_node = self.create_node(type=name)
            self.add_edge(source, temp_node)
            source = temp_node
        return source
    
    def get_node_info(self, name):
        node = None
        if isinstance(name, int):
            name = str(name)
        for n in self.G.nodes:
            if n.name == name:
                node = n
                break
        return node

    def print_prefix(self):
        for node in self.G.nodes:
            if node.type in [NodeType.PREFIX, NodeType.POSTFIX]:
                print(f"{node.name} in={len(self.preds(node))} out={len(self.succs(node))}")

    def hook_fn(self, node):
        def hook(module, input, _):
            a = FeatureReshapeHandler(module.__class__.__name__, module)
            b = a.handler(input[0])
            print(f"Hooked {node} with shape {b.shape}")
            self.intermediates[node] = b
            return None
        return hook

    def add_hooks(self):
        self.clear_hooks()

        for node in self.G.nodes:
            if node.type == NodeType.PREFIX: #or node.type == NodeType.SUM:
                print(f"Adding hook for {node.layer_name or node.name}")

                for succ in self.succs(node):
                    print(f"Trying {succ.layer_name} with type {succ.type}")

                    if succ.type == NodeType.MODULE:
                        # succ.color = 'green'
                        # add it to the first module
                        self.hooks.append(self.named_modules[succ.layer_name].register_forward_hook(self.hook_fn(node)))
                        break
                    elif node.type == NodeType.EMBEDDING:
                        def prehook(m, x, this_node=node, this_info=succ):
                            tensor = self.get_parameter(this_info['param']).data
                            tensor = tensor.flatten(0, len(x[0].shape)-2).transpose(1, 0).contiguous()
                            print(this_node)
                            self.intermediates[this_node.name] = tensor
                            return None
                    
                        module = self.get_module(succ.layer)
                        self.hooks.append(module.register_forward_pre_hook(prehook))
                        break
                    else:
                        raise RuntimeError(f"PREFIX node {node} had no module to attach to.")
            elif node.type == NodeType.POSTFIX:
                for pred in self.preds(node):
                    if pred.type == NodeType.MODULE:
                        # pred.color = 'red'
                        self.hooks.append(self.named_modules[pred.layer_name].register_forward_hook(self.hook_fn(node)))
                        break
                    elif node.type == NodeType.EMBEDDING:
                        raise RuntimeError("This shouldn't be hit for any reason")
                    else:
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

class TransformerEncoderGraph(ModelGraph):
    
    def __init__(self, model,
                 modules,
                 layer_name='', # for transformer
                 enc_prefix='encoder',
                 merge_type='ff_only',
                 num_layers=12,
                 num_heads=8,
                 qk=False,
                 name='bert',
                 classifier=False):
        super().__init__(model)
        
        self.layer_name = layer_name
        self.enc_prefix = enc_prefix
        self.merge_type = merge_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.modules = modules
        self.qk = qk
        self.name = name
        self.classifier = classifier 


    def add_layerblock_nodes(self, name_prefix, input_node, merge_type):
        # first half
        modules = self.modules
        # do attention block here
        residual = input_node
        value_node = self.add_nodes_from_sequence(name_prefix, [modules['v']], residual)
        if self.qk:
            key_node = self.add_nodes_from_sequence(name_prefix, [modules['k'], NodeType.POSTFIX], residual)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['q'], NodeType.POSTFIX, NodeType.SUM], residual)
        else:
            key_node = self.add_nodes_from_sequence(name_prefix, [modules['k']], residual)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['q'], NodeType.SUM], residual)
        self.add_edge(key_node, input_node) # add key to "SUM" - it is really just a product but same handler
        input_node = self.add_nodes_from_sequence(name_prefix, [NodeType.SUM], input_node) #sum (mult)node to outproj
        self.add_edge(value_node, input_node) #value node to sum (mult)
        
        if merge_type == 'ff_only':
            # add self attn out proj to dot prod, layer norm, sum residual
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn'], NodeType.SUM], 
                                                    input_node)
            # add & norm
            self.add_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)

        if merge_type == 'res_only':
            # add self attn out proj to dot prod, layer norm, sum residual
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn'], NodeType.SUM], 
                                                    input_node)
            # add & norm
            self.add_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)

        elif merge_type == 'ff+res':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get first residual vector from after self attn layer norm
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)

        elif merge_type == 'ff+attn':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)

        elif merge_type == 'attn_only':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)

        elif merge_type == 'res+attn':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)


        elif merge_type == 'all':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            # get first residual vector from after self attn layer norm
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_edge(residual, input_node)

        if merge_type in ['all', 'ff+res', 'res_only', 'res+attn']:
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln'], NodeType.POSTFIX], input_node=input_node)
        else:
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln']], input_node=input_node)
        return input_node

    def add_layer_nodes(self, layer_prefix, input_node, merge_type):
        source_node = input_node
        
        for layer_index in range(self.num_layers): # for graph visualization
        #for layer_index, layerblock in enumerate(self.get_module(name_prefix)):
            source_node = self.add_layerblock_nodes(layer_prefix+f'.{layer_index}', source_node, merge_type)        
        return source_node

    def graphify(self):
        modules = self.modules
        # keep input node
        input_node = self.create_node(type=NodeType.INPUT)
        # input_node -> emb_tok 
        emb_name = modules['emb']
        emb_node = self.create_node(type=NodeType.EMBEDDING, 
                                    layer_name=f'{self.enc_prefix}.{emb_name}'.strip('.'),
                                    param_name=f'{self.enc_prefix}.{emb_name}.weight'.strip('.'))
        self.add_edge(input_node, emb_node)

        # removing emb_pos node for now...
        input_node = self.add_nodes_from_sequence('', [modules['emb_ln']], emb_node) 
     
        if self.merge_type in ['all', 'ff+res', 'res_only']:
            #adding postfix to emb_ln, before xformer layers
            input_node = self.add_nodes_from_sequence(self.enc_prefix, [NodeType.POSTFIX], input_node)

        # layernorm_embedding -> xformer layers
        input_node = self.add_layer_nodes(f'{self.layer_name}', input_node, self.merge_type)
                
        # xformer layers -> dense -> layernorm -> output
        #if self.name == 'bert' and self.classifier == False:
        #    dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['transform.dense', 'transform.LayerNorm', NodeType.PREFIX, 'decoder'], input_node)
        #    output_node = self.create_node(type=NodeType.OUTPUT)
        #    self.add_edge(dense_node, output_node)
        #elif self.name == 'bert' and self.classifier == True:
        #    pool_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['pooler']], input_node)
        #    class_node = self.add_nodes_from_sequence('', [NodeType.PREFIX, modules['classifier']], pool_node)
        #    output_node = self.create_node(type=NodeType.OUTPUT)
        #    self.add_edge(class_node, output_node)
        #elif self.name == 'roberta':
        #    #dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['dense', NodeType.PREFIX, 'out_proj'], input_node)
        #    output_node = self.create_node(type=NodeType.OUTPUT)
        #    self.add_edge(input_node, output_node)       

        if self.name == 'bert':
           #dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['dense', NodeType.PREFIX, 'out_proj'], input_node)
           output_node = self.create_node(type=NodeType.OUTPUT)
           self.add_edge(input_node, output_node)
        
        return self

    
def bert(model, merge_type='all', qk=False, classifier=False):
    modules = {'emb': 'embeddings.word_embeddings',
     'emb_pos': 'embeddings.position_embeddings',
     'emb_tok_type': 'embeddings.token_type_embeddings',
     'emb_ln': 'embeddings.LayerNorm',
     'q': 'attention.self.query',
     'k': 'attention.self.key',
     'v': 'attention.self.value',
     'lin_attn': 'attention.output.dense',
     'attn_ln': 'attention.output.LayerNorm',
     'fc1': 'intermediate.dense',
     'fc2': 'output.dense',
     'final_ln': 'output.LayerNorm',
     'head_pref': 'cls.predictions',
     'pooler': 'pooler.dense',
     'classifier': 'classifier'}
    return TransformerEncoderGraph(model, 
                                   modules,
                                   layer_name='encoder.layer', 
                                   enc_prefix='encoder',
                                   merge_type=merge_type,
                                   num_layers=12,
                                   num_heads=12,
                                   qk=qk,
                                   name='bert',
                                   classifier=classifier)


def interpolate_color(start_color, end_color, steps):
    """Interpolate colors from start to end in RGB space over a given number of steps."""
    start_rgb = [int(start_color[i:i+2], 16) for i in (1, 3, 5)]
    end_rgb = [int(end_color[i:i+2], 16) for i in (1, 3, 5)]
    
    colors = []
    for step in range(steps):
        interpolated = [start + (end - start) * step / (steps - 1) for start, end in zip(start_rgb, end_rgb)]
        colors.append('#' + ''.join(f'{int(round(c)):02x}' for c in interpolated))
    return colors

# Start and end colors in hex format
start_color_hex = "#add8e6"  # Light blue
end_color_hex = "#0000ff" 

def compute_transformations(graph, nodes, res):

        merges = {}
        unmerges = {}
           

        global_res_merge= None
        global_res_unmerge = None

        special_cases_names = ['final_ln', 'attn_ln', 'emb_ln', 'q', 'k']
        special_cases_nodes = [graph.modules[name] for name in special_cases_names]
        qk_nodes = [graph.modules[name] for name in ['q', 'k']]
        print('qk nodes', qk_nodes)

        cost_dict = {}
        
        qk_flag = False
        if graph.qk == True:
            qk_flag = True
            #for i in range(graph.num_layers):
            #    nodes.remove(f'qk{i}')
        # nodes.sort() # what the fuck is going on here?
        print('computing corrs')
        # corrs = compute_metric_corrs(nodes, res=res, no_corr=no_corr, qk=qk_flag)

        # save all corrs to file to look at them. 
        # breakpoint()
        # with open(f'corrs.pt', 'wb+') as corrs_out:
        #     torch.save(corrs, corrs_out)
        
        # corrs has all nonres nodes & the one res node. Unless this is sep, then it has all nodes

        for node in tqdm(nodes, desc="Computing transformations: "):
            prev_node_layer = graph.get_node_info(int(node.name)-1).layer_name
            # skip metrics associated with residuals and qk if qk is true
            correlation_matrix = None
            print(f"Print prev node layer: {prev_node_layer}")

            # Boolean algebra here fucko
            if not prev_node_layer or not any([name in prev_node_layer for name in special_cases_nodes]):
                node.color = 'violet'
            elif any([name in prev_node_layer for name in qk_nodes]):
                node.color = 'black'
            else:
                node.color = 'blue'

        return


if __name__ == "__main__":
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    g = bert(model, merge_type='all', qk=True, classifier=False)
    g = g.graphify()

    g.add_hooks()
    
    x = torch.ones(1, 128).long()
    # convert x to int
    intermediates = g.compute_intermediates(x)
    print("len intermediates", len(intermediates))

    nodes = []

    for k, v in intermediates.items():
        print(f"{k} -> :{v.shape} {k.layer_name}")
        # k.color = 'blue'
        nodes.append(k)

    def separate_res_nodes(nodes, graph):
        resnodes = []
        non_resnodes = []
        for node in nodes:
            if node.type == NodeType.POSTFIX:
                print(f"Node: {node.name}")
                prev_node_info = graph.get_node_info(int(node.name) - 1).layer_name
                if (graph.modules['q'] in prev_node_info) or (graph.modules['k'] in prev_node_info):
                    #non_resnodes.append(node) # this is a qk node
                    continue
                else:
                    resnodes.append(node) # all res keys are postfixes by design
            else:
                non_resnodes.append(node)
        return resnodes, non_resnodes

    resnodes, non_resnodes = separate_res_nodes(nodes, g)
    #for node in resnodes:
    #    # node.color = 'black'
    #    print(f"Residual node: {node}")

    #for node in non_resnodes:
    #    # node.color = 'gray'
    #    print(f"Non-residual node: {node}")

    compute_transformations(g, nodes, 'all')

    ## color = interpolate_color(start_color_hex, end_color_hex, len(nodes))

    ##for node, c in zip(nodes, color):
    ##    print(node)
    ##    node.color = c


    # traversal code

    #qk_flag = False

    #for i, node in enumerate(nodes):
    #    node.color = 'red'
    #    node.traversal_number = i
    #    count = 0
    #    # gross code
    #    preds = g.G.predecessors(node)
    #    info = preds[0]
    #    info.color = 'red'
    #    info.traversal_number = i
    #    # self attention merging, and self attention out unmerging 
    #    if info.type == NodeType.SUM:
    #        print(f'merging MHA : {info}')
    #        # apply merges to k,q,v matrices
    #        sum_preds = g.G.predecessors(preds[0])
    #        for sum_pred in sum_preds:
    #            sum_pred.color = 'red'
    #        # check if q,k junction or v matrix
    #        for sum_pred in sum_preds:
    #            sum_pred.traversal_number = i
    #            
    #            info = sum_pred
    #            if info.type == NodeType.SUM:
    #                sum_pred.color = 'red'
    #                if qk_flag == False:
    #                    second_sum_preds = g.G.predecessors(sum_pred)
    #                    # merge q & k 
    #                    for second_sum_pred in second_sum_preds:
    #                        second_sum_pred.color = 'red'
    #                        second_sum_pred.traversal_number = i
    #            elif 'v_proj' in info.layer_name or 'value' in info.layer_name:
    #                # merge v
    #                # do something with sum pred
    #                # self.merge_node(sum_pred, merger)
    #                sum_pred.color = 'green'
    #        break

    #        # color the successor node

        #    succ = merger.graph.succs(node)[0]
        #    self.unmerge_node(succ, merger)
        #elif contains_name(info['layer'], qk_nodes) and qk_flag == True:
        #    print('merging qk')
        #    self.merge_node(preds[0], merger)

        #elif 'self_attn_layer_norm' in info['layer'] or 'attention.output.LayerNorm' in info['layer']:
        #    print('merging self-attn res')
        #    # apply merge to ln
        #    module = merger.graph.get_module(info['layer'])
        #    parameter_names = ['weight', 'bias']
        #    for parameter_name in parameter_names:
        #        parameter = getattr(module, parameter_name)
        #        parameter.data = merger.merge @ parameter

        #    # apply merges to the self.attn out proj
        #    sum = merger.graph.preds(preds[0])[0]
        #    out_proj = merger.graph.preds(sum)[0]
        #    self.merge_node(out_proj, merger)

        #    # unmerge the ff1 module 
        #    ff1 = merger.graph.succs(node)[0]
        #    self.unmerge_node(ff1, merger)

        #elif 'final_layer_norm' in info['layer'] or 'layernorm_embedding' in info['layer'] or 'output.LayerNorm' in info['layer'] or 'embeddings.LayerNorm' in info['layer']:
        #    print('merging final res')
        #    # apply merge to ln
        #    module = merger.graph.get_module(info['layer'])
        #    parameter_names = ['weight', 'bias']
        #    for parameter_name in parameter_names:
        #        parameter = getattr(module, parameter_name)
        #        parameter.data = merger.merge @ parameter

        #    sum = merger.graph.preds(preds[0])[0]
        #    info = merger.graph.get_node_info(sum)
        #    if info['type'] == NodeType.SUM:
        #        ff2 = merger.graph.preds(sum)[0]
        #        self.merge_node(ff2, merger)
        #    else:
        #        # this is emb node then
        #        if final_merger == None and count == 1:
        #            final_merger = merger
        #        if merger.graph.enc_prefix == 'bert':
        #            # bert has special token type embedding that must be merged too
        #            emb_tok_suff = merger.graph.modules['emb_tok_type']
        #            emb_tok_name = f'{merger.graph.enc_prefix}.{emb_tok_suff}'
        #            emb_tok_mod = merger.graph.get_module(emb_tok_name)
        #            emb_tok_mod.weight.data = (merger.merge @ (emb_tok_mod.weight).T).T 

        #        # grabbing naming vars
        #        emb_suff = merger.graph.modules['emb']
        #        emb_pos_suff = merger.graph.modules['emb_pos']
        #        emb_name = f'{merger.graph.enc_prefix}.{emb_suff}'
        #        emb_pos_name = f'{merger.graph.enc_prefix}.{emb_pos_suff}'

        #        # merger emb &  emb_pos
        #        emb = merger.graph.get_module(emb_name)
        #        emb_pos = merger.graph.get_module(emb_pos_name)
        #        emb.weight.data = (merger.merge @ (emb.weight).T).T
        #        emb_pos.weight.data = (merger.merge @ (emb_pos.weight).T).T 

        #    # this unmerges w_k, w_q, w_v
        #    succs = merger.graph.succs(node)
        #    if len(succs) > 1:
        #        for succ in succs:
        #            info = merger.graph.get_node_info(succ)
        #            if info['type'] != NodeType.SUM:
        #                self.unmerge_node(succ, merger)
        #    else:
        #        # in this case, we have the second to last node
        #        # separate case for mnli & camembert due to head names
        #        # first we check if model is bert and unmerge the lm head 
        #        if 'cls.predictions.transform.dense' in merger.graph.named_modules:
        #            module = merger.graph.get_module('cls.predictions.transform.dense') 
        #            module.weight.data = module.weight @ merger.unmerge

        #        elif 'bert.pooler.dense' in merger.graph.named_modules:
        #            module = merger.graph.get_module('bert.pooler.dense') 
        #            module.weight.data = module.weight @ merger.unmerge
        #        elif len(merger.graph.model.classification_heads.keys()) != 0:
        #            if 'classification_heads.mnli.dense' in merger.graph.named_modules:
        #                module = merger.graph.get_module('classification_heads.mnli.dense')
        #                module.weight.data = module.weight @ merger.unmerge
        #            elif 'classification_heads.sentence_classification_head.dense' in merger.graph.named_modules:
        #                module = merger.graph.get_module('classification_heads.sentence_classification_head.dense')
        #                module.weight.data = module.weight @ merger.unmerge
        #        # if has no classification heads, it uses lm heads instead, and is a roberta model
        #        # unmerge this, but in the actual eval of wsc, need to fix forward pass, but this is the minimum needed to
        #        # store the correct weights
        #        else:
        #            module = merger.graph.get_module('encoder.lm_head.dense')
        #            module.weight.data = module.weight @ merger.unmerge

        ## apply merge to fc1 & unmerge fc2
        #elif 'fc1' in info['layer'] or 'intermediate.dense' in info['layer']:
        #    print('merging ff')
        #    # apply merges to the fc1 layer
        #    module = merger.graph.get_module(info['layer'])
        #    self.merge_node(preds[0], merger)

        #    # apply unmerge to fc2 layer
        #    succ = merger.graph.succs(node)[0]
        #    self.unmerge_node(succ, merger)

        #elif 'transform.LayerNorm' in info['layer'] and merge_cls:
        #    if final_merger == None and count == 1: # count ensures this is 2nd model merger being saved
        #        final_merger = merger

        #    print('merging lm head')
        #    # apply merge to layernorm 
        #    module = merger.graph.get_module(info['layer'])
        #    parameter_names = ['weight', 'bias']
        #    for parameter_name in parameter_names:
        #        parameter = getattr(module, parameter_name)
        #        parameter.data = merger.merge @ parameter

        #    # merge dense
        #    pred = merger.graph.preds(preds[0])[0]
        #    self.merge_node(pred, merger)

        #elif 'pooler' in info['layer'] and merge_cls:
        #    print('merging class head')
        #    # merge pooler weight
        #    self.merge_node(preds[0], merger)
        #    # get cls node & unmerge
        #    succ = merger.graph.succs(node)[0]
        #    self.unmerge_node(succ, merger)
        #count += 1

    print("Drawing graph")
    diagram = g.G.draw()
    diagram.render('bert_traversal', format='png')