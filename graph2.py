import torch
import networkx as nx
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from graphviz import Digraph


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

            t = (node.name, node.layer_name)
            color = {}
            if node.type == NodeType.MODULE:
                pass
            elif node.type == NodeType.PREFIX:
                t = (node.name, "Prefix")
            elif node.type == NodeType.POSTFIX:
                t = (node.name, "Postfix")
            elif node.type == NodeType.SUM:
                t = (node.name, "Sum")
            else:
                t = (node.name, node.name)
            if node.color is not None:
                color = {'color': node.color, 'style': 'filled'}
            print(f"Adding node {t} with color {color}")
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
                temp_node = self.create_node(layer_name=name_prefix + sep + name)
            else:
                temp_node = self.create_node(type=name)
            self.add_edge(source, temp_node)
            source = temp_node
        return source

    def print_prefix(self):
        for node in self.G.nodes:
            if node.type in [NodeType.PREFIX, NodeType.POSTFIX]:
                print(f"{node.name} in={len(self.preds(node))} out={len(self.succs(node))}")

    def hook_fn(self, node):
        def hook(module, input, _):
            a = FeatureReshapeHandler(module.__class__.__name__, module)
            b = a.handler(input[0])
            self.intermediates[node] = b
            print(f"Hooked {node.name} with shape {b.shape}")
            return None
        return hook

    def add_hooks(self):
        self.clear_hooks()

        for node in self.G.nodes:
            if node.type == NodeType.PREFIX: #or node.type == NodeType.SUM:

                for succ in self.succs(node):
                    print(f"Trying {succ.layer_name} with type {succ.type}")

                    if succ.type == NodeType.MODULE:
                        # add it to the first module
                        self.hooks.append(self.named_modules[succ.layer_name].register_forward_hook(self.hook_fn(node)))
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
        input_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['emb_ln']], emb_node) 
     
        if self.merge_type in ['all', 'ff+res', 'res_only']:
            #adding postfix to emb_ln, before xformer layers
            input_node = self.add_nodes_from_sequence(self.enc_prefix, [NodeType.POSTFIX], input_node)

        # layernorm_embedding -> xformer layers
        input_node = self.add_layer_nodes(f'{self.layer_name}', input_node, self.merge_type)
                
        # xformer layers -> dense -> layernorm -> output
        if self.name == 'bert' and self.classifier == False:
            dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['transform.dense', 'transform.LayerNorm', NodeType.PREFIX, 'decoder'], input_node)
            output_node = self.create_node(type=NodeType.OUTPUT)
            self.add_edge(dense_node, output_node)
        elif self.name == 'bert' and self.classifier == True:
            pool_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['pooler']], input_node)
            class_node = self.add_nodes_from_sequence('', [NodeType.PREFIX, modules['classifier']], pool_node)
            output_node = self.create_node(type=NodeType.OUTPUT)
            self.add_edge(class_node, output_node)
        elif self.name == 'roberta':
            #dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['dense', NodeType.PREFIX, 'out_proj'], input_node)
            output_node = self.create_node(type=NodeType.OUTPUT)
            self.add_edge(input_node, output_node)       
        
        return self

    
def bert(model, merge_type='ff_only', qk=False, classifier=False):
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
                                   layer_name='bert.encoder.layer', 
                                   enc_prefix='bert',
                                   merge_type=merge_type,
                                   num_layers=12,
                                   num_heads=12,
                                   qk=qk,
                                   name='bert',
                                   classifier=classifier)

if __name__ == "__main__":
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    g = bert(model, merge_type='ff_only', qk=False, classifier=False)
    diagram = g.graphify().G.draw()
    diagram.render('bert', format='png')
