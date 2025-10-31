from itertools import product
import warnings

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy


def sample_rules(v, n, m, s, L, seed=42):
    """
    Sample random rules for a random hierarchy model.

    Args:
        v: The number of values each variable can take (vocabulary size, int).
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        seed: Seed for generating the rules.

    Returns:
        A dictionary containing the rules for each level of the hierarchy.
    """
    random.seed(seed)
    tuples = list(product(*[range(v) for _ in range(s)]))

    rules = {}
    rules[0] = torch.tensor(random.sample(tuples, n * m)).reshape(n, m, -1)
    for i in range(1, L):
        rules[i] = torch.tensor(random.sample(tuples, v * m)).reshape(v, m, -1)

    return rules


def sample_data_from_generator_classes(g, y, rules, return_tree_structure=False) -> tuple[torch.Tensor, torch.Tensor] | tuple[dict, torch.Tensor]:
    """
    Create data of the Random Hierarchy Model starting from its rules, a seed and a set of class labels.

    Args:
        g: A torch.Generator object.
        y: A tensor of size [batch_size, 1] containing the class labels.
        rules: A dictionary containing the rules for each level of the hierarchy.
        return_tree_structure: If True, return the tree structure of the hierarchy as a dictionary.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    labels = copy.deepcopy(y)

    if return_tree_structure:
        x_st = (
            {}
        )  # Initialize the dictionary to store the values and messages for each variable in the hierarchy
        x_st[0] = y
        for i in range(L):  # Loop over the levels of the hierarchy
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=g
            )  # Choose a random rule for each variable in the current level
            x_st[i + 1] = rules[i][x_st[i], chosen_rule].flatten(
                start_dim=1
            )  # Apply the chosen rule to each variable in the current level
        return x_st, labels
    else:
        x = y
        for i in range(L):
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x.shape, generator=g
            )
            x = rules[i][x, chosen_rule].flatten(start_dim=1)
        return x, labels


def sample_with_replacement(train_size, test_size, seed_sample, rules) -> tuple[torch.Tensor, torch.Tensor]:

    n = rules[0].shape[0]  # Number of classes

    if train_size == -1:
        warnings.warn(
            "Whole dataset (train_size=-1) not available with replacement! Using train_size=1e6.",
            RuntimeWarning,
        )
        train_size = 1000000

    g = torch.Generator()
    g.manual_seed(seed_sample)

    y = torch.randint(low=0, high=n, size=(train_size + test_size,), generator=g)
    features, labels = sample_data_from_generator_classes(g, y, rules)
    
    # Ensure features is a tensor
    if isinstance(features, dict):
        raise TypeError(f"sample_data_from_generator_classes returned dict instead of Tensor for features")
    
    return features, labels


def sample_synonyms_from_generator_classes(g, y, rules, syn_g, syn_layer, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from its rules, a seed and a set of class labels.

    Args:
        g: A torch.Generator object.
        y: A tensor of size [batch_size, 1] containing the class labels.
        rules: A dictionary containing the rules for each level of the hierarchy.
        syn_g: A torch.Generator object for sampling synonyms.
        syn_layer: The layer at which synonyms are sampled.
        return_tree_structure: If True, return the tree structure of the hierarchy as a dictionary.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    labels = copy.deepcopy(y)

    if return_tree_structure:
        x_st = (
            {}
        )  # Initialize the dictionary to store the values and messages for each variable in the hierarchy
        x_st[0] = y
        for i in range(L):  # Loop over the levels of the hierarchy
            if i < syn_layer:
                chosen_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=g
                )  # Choose a random rule for each variable in the current level
                x_st[i + 1] = rules[i][x_st[i], chosen_rule].flatten(
                    start_dim=1
                )  # Apply the chosen rule to each variable in the current level
            elif i == syn_layer:
                chosen_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=g
                ) 
                #choose another set of rules that is not the same as the chosen rule
                synonymic_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=syn_g
                )
                synonymic_rule = torch.where(synonymic_rule == chosen_rule, (synonymic_rule + 1) % rules[i].shape[1], synonymic_rule)
                x_st[i + 1] = rules[i][x_st[i], synonymic_rule].flatten(
                start_dim=1
                )  
            else:
                synonymic_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=syn_g
                )
                x_st[i + 1] = rules[i][x_st[i], synonymic_rule].flatten(
                start_dim=1
                )
        return x_st, labels
    else:
        x = y
        for i in range(L):
            if i < syn_layer:
                chosen_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x.shape, generator=g
                )
                x = rules[i][x, chosen_rule].flatten(start_dim=1)
            elif i == syn_layer:
                chosen_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x.shape, generator=g
                ) 
                synonymic_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x.shape, generator=syn_g
                )
                synonymic_rule = torch.where(synonymic_rule == chosen_rule, (synonymic_rule + 1) % rules[i].shape[1], synonymic_rule)
                x = rules[i][x, synonymic_rule].flatten(
                start_dim=1
                )
            else:
                synonymic_rule = torch.randint(
                    low=0, high=rules[i].shape[1], size=x.shape, generator=syn_g
                )
                x = rules[i][x, synonymic_rule].flatten(
                start_dim=1
                )
        return x, labels
    

def sample_data_and_synonyms_with_replacement(sample_size, seed_sample, rules, synonims_layer, synonims_seed, return_tree_structure=False):

    # Check if synonyms layer is just a number
    if isinstance(synonims_layer, int):
        synonims_layer = [synonims_layer]

    n = rules[0].shape[0]  # Number of classes
    syn_x = {}

    g = torch.Generator()
    g.manual_seed(seed_sample)
    y = torch.randint(low=0, high=n, size=(sample_size,), generator=g)
    samples, labels = sample_data_from_generator_classes(g, y, rules, return_tree_structure)
    syn_x['data'] = samples

    for syn_layer in synonims_layer:
        g = torch.Generator()
        g.manual_seed(seed_sample)
        syn_g = torch.Generator()
        syn_g.manual_seed(synonims_seed)
        y = torch.randint(low=0, high=n, size=(sample_size,), generator=g)
        # samples, labels = sample_data_from_generator_classes(g, y, rules)
        synonims, labels = sample_synonyms_from_generator_classes(g, y, rules, syn_g, syn_layer, return_tree_structure)
        syn_x[syn_layer] = synonims

    return syn_x, labels


def sample_data_from_indeces(samples, rules, n, m, s, L):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    max_data = n * m ** ((s**L - 1) // (s - 1))
    data_per_hl = max_data // n  # div by num_classes to get number of data per class

    high_level = samples.div(
        data_per_hl, rounding_mode="floor"
    )  # div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl  # compute remainder (run in range(data_per_hl))

    labels = high_level  # labels are the classes (features of highest level)
    features = labels  # init input features as labels (rep. size 1)
    size = 1

    for l in range(L):

        choices = m ** (size)
        data_per_hl = (
            data_per_hl // choices
        )  # div by num_choices to get number of data per high-level feature

        high_level = low_level.div(
            data_per_hl, rounding_mode="floor"
        )  # div by data_per_hl to get high-level feature index (1 index in range(m**size))
        high_level = dec2base(
            high_level, m, length=size
        ).squeeze()  # convert to base m (size indices in range(m), squeeze needed if index already in base m)

        features = rules[l][
            features, high_level
        ]  # apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
        features = features.flatten(
            start_dim=1
        )  # flatten to tensor of size (batch_size, size*s)
        size *= s  # rep. size increases by s at each level

        low_level = (
            low_level % data_per_hl
        )  # compute remainder (run in range(data_per_hl))

    return features, labels


def sample_without_replacement(max_data, train_size, test_size, seed_sample, rules, allow_replacement_failover=False) -> tuple[torch.Tensor, torch.Tensor]:
    
    L = len(rules)  # Number of levels in the hierarchy
    n = rules[0].shape[0]  # Number of classes
    m = rules[0].shape[1]  # Number of synonymic lower-level representations
    s = rules[0].shape[2]  # Size of lower-level representations
    if(allow_replacement_failover and max_data>2**63-1):
        return sample_with_replacement(train_size, test_size, seed_sample, rules)
    assert(max_data < 2**63-1), "max_data too large -- it must be representable as a 64-bit signed integer."
    if train_size == -1:
        samples = torch.arange(max_data)
    else:
        test_size = min(test_size, max_data - train_size)

        random.seed(seed_sample)
        samples = torch.tensor(random.sample(range(max_data), train_size + test_size))

    features, labels = sample_data_from_indeces(samples, rules, n, m, s, L)

    return features, labels

def infer_model_parameters_from_rules(rules):
    """
    Infers model parameters from a given set of rules.
    Args:
        rules (list of torch.Tensor): A list or dictionary of tensors representing the rules. 
                                      Each tensor should have the same shape.
    Returns:
        dict: A dictionary containing the inferred parameters:
            - 'm' (int): The number of synonyms.
            - 's' (int): The branching ratio of the RHM.
            - 'L' (int): The number of layers of the hierarchy, total dimension is s**d.
            - 'n' (int): The number of top-level classes.
            - 'v' (int): The inferred vocabulary size, from the largest symbol in any of the rules.
    """
    params = {
        'm': rules[0].shape[1],
        's': rules[0].shape[2],
        'L': len(rules),
        'n': rules[0].shape[0],
        'v': 1+max([torch.max(rules[i]).item() for i in range(1,len(rules))]), #there's no direct way to infer v from the rules
    }
    return params 

def convert_features_to_one_hot(features:torch.Tensor, v:int):
    """
    Converts a tensor of features to a one-hot encoded tensor.
    Args:
        features (torch.tensor): The input tensor containing feature indices.
        v (int): The number of classes for one-hot encoding.
    Returns:
        torch.tensor: A one-hot encoded tensor with the same shape as the input tensor, 
                        but with an additional dimension for the one-hot encoding.
    """
    return F.one_hot(features, v).transpose(-2,-1).float()


def compute_pmax(n:int, m:int, s:int, L:int):
    return n * m ** ((s**L - 1) // (s - 1))



class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """
    features: torch.Tensor
    labels: torch.Tensor

    def __init__(
        self,
        num_features=8,
        num_classes=2,
        num_synonyms=2,
        tuple_size=2,  # size of the low-level representations
        num_layers=2,
        seed_rules=0,
        seed_sample=1,
        train_size=-1,
        test_size=0,
        input_format="onehot",
        whitening=0,
        transform=None,
        replacement=False,
        upload_rules_from_path=None,
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size

        if upload_rules_from_path is not None:
            rules = torch.load(upload_rules_from_path)
            assert rules[0].shape[0] == num_classes, "Rules shape does not match"
            assert rules[0].shape[1] == num_synonyms, "Rules shape does not match"
            assert rules[0].shape[2] == tuple_size, "Rules shape does not match"
            assert len(rules) == num_layers, "Rules shape does not match"
        else:
            rules = sample_rules(
                num_features,
                num_classes,
                num_synonyms,
                tuple_size,
                num_layers,
                seed=seed_rules,
            )

        self.rules = rules

        max_data = num_classes * num_synonyms ** (
            (tuple_size**num_layers - 1) // (tuple_size - 1)
        )
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > 1e19 and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:
            self.features, self.labels = sample_without_replacement(
                max_data, train_size, test_size, seed_sample, rules
            )
        else:
            self.features, self.labels = sample_with_replacement(
                train_size, test_size, seed_sample, rules
            )

        if "onehot" not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        # TODO: implement one-hot encoding of s-tuples
        if "onehot" in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=(
                    num_features
                    if "tuples" not in input_format
                    else num_features**tuple_size
                ),
            ).float()

            if whitening:

                inv_sqrt_norm = (1.0 - 1.0 / num_features) ** -0.5
                self.features = (self.features - 1.0 / num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif "long" in input_format:
            self.features = self.features.long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
                idx: sample index

        Returns:
            Feature-label pairs at index
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

    def get_rules(self):
        return self.rules



def dec2bin(x, bits=None):
    """
    Convert integers to binary.
    
    Args:
            n: The numbers to convert (tensor of size [*]).
         bits: The length of the representation.
    Returns:
        A tensor (size [*, bits]) with the binary representations.
    """
    if bits is None:
        bits = (x.max() + 1).log2().ceil().item()
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def dec2base(n, b, length=None):
    """
    Convert integers into a different base.
    
    Args:
            n: The numbers to convert (tensor of size [*]).
            b: The base (integer).
       length: The length of the representation.
    Returns:
        A tensor (size [*, length]) containing the input numbers in the new base.
    """
    digits = []
    while n.sum():
        digits.append(n % b)
        n = n.div(b, rounding_mode='floor')
    if length:
        assert len(digits) <= length, "Length required is too small to represent input numbers!"
        digits += [torch.zeros(len(n), dtype=torch.int)] * (length - len(digits))
    return torch.stack(digits[::-1]).t()