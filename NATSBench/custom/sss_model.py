import sys
sys.path.append("..") # Adds higher directory to python modules path

from typing import List, Text, Any
import random, torch
import torch.nn as nn

from xautodl.models.cell_operations import ResNetBasicblock
from xautodl.models.cell_infers.cells import InferCell
from xautodl.models.cell_searchs import CellStructure

class DynamicShapeTinyNet(nn.Module):
    def __init__(self, channels: List[int], genotype: Any, num_classes: int):
        super(DynamicShapeTinyNet, self).__init__()
        self._channels = channels
        if len(channels) % 3 != 2:
            raise ValueError("invalid number of layers : {:}".format(len(channels)))
        self._num_stage = N = len(channels) // 3

        genotype = CellStructure.str2structure(genotype)

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )

        # layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        c_prev = channels[0]
        self.cells = nn.ModuleList()
        for index, (c_curr, reduction) in enumerate(zip(channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(c_prev, c_curr, 2, True)
            else:
                cell = InferCell(genotype, c_prev, c_curr, 1)
            self.cells.append(cell)
            c_prev = cell.out_dim
        self._num_layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(c_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_channels}, N={_num_stage}, L={_num_layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def extract_cell_features(self, inputs):
        cell_features = []

        feature = self.stem(inputs)
        if feature.requires_grad:
            feature.retain_grad()
            cell_features.append(feature)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            if feature.requires_grad:
                feature.retain_grad()
            cell_features.append(feature)

        return cell_features

    def extract_cell_features_and_logits(self, inputs):
        cell_features = []

        feature = self.stem(inputs)
        if feature.requires_grad:
            feature.retain_grad()
            cell_features.append(feature)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            if feature.requires_grad:
                feature.retain_grad()
            cell_features.append(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return cell_features, logits

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits




########
# import sys
# sys.path.append("..") # Adds higher directory to python modules path

# from typing import List, Text, Any
# import random, torch
# import torch.nn as nn

# from xautodl.models.cell_operations import ResNetBasicblock
# from xautodl.models.cell_infers.cells import InferCell
# from xautodl.models.shape_searchs.SoftSelect import select2withP, ChannelWiseInter
# from xautodl.models.cell_searchs import CellStructure

# def get_cell_based_tiny_net(config):
#     if isinstance(config, dict):
#         config = dict2config(config, None)  # to support the argument being a dict
#     super_type = getattr(config, "super_type", "basic")
#     if super_type == "search-shape":
#         genotype = CellStructure.str2structure(config.genotype)
#         return GenericNAS301Model(
#             config.candidate_Cs,
#             config.max_num_Cs,
#             genotype,
#             config.num_classes,
#             config.affine,
#             config.track_running_stats,
#         )


# class GenericNAS301Model(nn.Module):
#     def __init__(
#         self,
#         candidate_Cs: List[int],
#         max_num_Cs: int,
#         genotype: Any,
#         num_classes: int,
#         affine: bool,
#         track_running_stats: bool,
#     ):
#         super(GenericNAS301Model, self).__init__()
#         self._max_num_Cs = max_num_Cs
#         self._candidate_Cs = candidate_Cs
#         if max_num_Cs % 3 != 2:
#             raise ValueError("invalid number of layers : {:}".format(max_num_Cs))
#         self._num_stage = N = max_num_Cs // 3
#         self._max_C = max(candidate_Cs)

#         stem = nn.Sequential(
#             nn.Conv2d(3, self._max_C, kernel_size=3, padding=1, bias=not affine),
#             nn.BatchNorm2d(
#                 self._max_C, affine=affine, track_running_stats=track_running_stats
#             ),
#         )

#         layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

#         c_prev = self._max_C
#         self._cells = nn.ModuleList()
#         self._cells.append(stem)
#         for index, reduction in enumerate(layer_reductions):
#             if reduction:
#                 cell = ResNetBasicblock(c_prev, self._max_C, 2, True)
#             else:
#                 cell = InferCell(
#                     genotype, c_prev, self._max_C, 1, affine, track_running_stats
#                 )
#             self._cells.append(cell)
#             c_prev = cell.out_dim
#         self._num_layer = len(self._cells)

#         self.lastact = nn.Sequential(
#             nn.BatchNorm2d(
#                 c_prev, affine=affine, track_running_stats=track_running_stats
#             ),
#             nn.ReLU(inplace=True),
#         )
#         self.global_pooling = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(c_prev, num_classes)
#         # algorithm related
#         self._algo = None

#     def set_algo(self, algo: Text):
#         # used for searching
#         assert self._algo is None, "This functioin can only be called once."
#         assert algo in ["training-free"], "invalid algo : {:}".format(
#             algo
#         )
#         self._algo = algo
#         self.register_buffer(
#             "_masks", torch.zeros(len(self._candidate_Cs), max(self._candidate_Cs))
#         )
#         for i in range(len(self._candidate_Cs)):
#             self._masks.data[i, : self._candidate_Cs[i]] = 1
#         # random init
#         self.register_buffer(
#             "layer_mask_indices", torch.zeros(self._max_num_Cs, dtype=torch.int32)
#         )
#         for i in range(self._max_num_Cs):
#             self.layer_mask_indices[i] = random.randint(0, len(self._candidate_Cs) - 1)

#     @property
#     def weights(self):
#         xlist = list(self._cells.parameters())
#         xlist += list(self.lastact.parameters())
#         xlist += list(self.global_pooling.parameters())
#         xlist += list(self.classifier.parameters())
#         return xlist

#     def str_to_genotype(self, s):
#         c_list = s.split(":")
#         assert len(c_list) == self._max_num_Cs
#         for i, c in enumerate(c_list):
#             c = int(c)
#             assert c in self._candidate_Cs
#             self.layer_mask_indices[i] = self._candidate_Cs.index(c)

#     def random_genotype(self, set_cache):
#         if set_cache:
#             for i in range(self._max_num_Cs):
#                 self.layer_mask_indices[i] = random.randint(0, len(self._candidate_Cs) - 1)
#             return self.genotype
#         else:
#             cs = []
#             for i in range(self._max_num_Cs):
#                 index = random.randint(0, len(self._candidate_Cs) - 1)
#                 cs.append(str(self._candidate_Cs[index]))
#             return ":".join(cs)

#     @property
#     def genotype(self):
#         cs = []
#         for i in range(self._max_num_Cs):
#             with torch.no_grad():
#                 index = self.layer_mask_indices[i]
#                 cs.append(str(self._candidate_Cs[index]))
#         return ":".join(cs)

#     def get_message(self) -> Text:
#         string = self.extra_repr()
#         for i, cell in enumerate(self._cells):
#             string += "\n {:02d}/{:02d} :: {:}".format(
#                 i, len(self._cells), cell.extra_repr()
#             )
#         return string

#     def extra_repr(self):
#         return "{name}(candidates={_candidate_Cs}, num={_max_num_Cs}, N={_num_stage}, L={_num_layer})".format(
#             name=self.__class__.__name__, **self.__dict__
#         )

#     def extract_cell_features(self, inputs):
#         cell_features = []
#         masks = []

#         feature = inputs

#         for i, cell in enumerate(self._cells):
#             feature = cell(feature)
#             # apply different searching algorithms
#             idx = max(0, i - 1)
#             mask = self._masks[self.layer_mask_indices[idx]]
#             feature = feature * mask.view(1, -1, 1, 1)
            
#             if feature.requires_grad:
#                 feature.retain_grad()
#             cell_features.append(feature)
#             masks.append(mask)
            
#         return cell_features, masks

#     def forward(self, inputs):
#         feature = inputs

#         for i, cell in enumerate(self._cells):
#             feature = cell(feature)
#             # apply different searching algorithms
#             idx = max(0, i - 1)
#             mask = self._masks[self.layer_mask_indices[idx]]
#             feature = feature * mask.view(1, -1, 1, 1)

#         out = self.lastact(feature)
#         out = self.global_pooling(out)
#         out = out.view(out.size(0), -1)
#         logits = self.classifier(out)

#         return out, logits
