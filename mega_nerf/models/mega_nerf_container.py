from typing import List

import torch
from torch import nn


class MegaNeRFContainer(nn.Module):
    def __init__(self, sub_modules: List[nn.Module], bg_sub_modules: List[nn.Module], image_embeddings: List[nn.Module],
                 centroids: torch.Tensor, grid_dim: torch.Tensor, min_position: torch.Tensor,
                 max_position: torch.Tensor, need_viewdir: bool):
        super(MegaNeRFContainer, self).__init__()

        for i, sub_module in enumerate(sub_modules):
            setattr(self, 'sub_module_{}'.format(i), sub_module)

        for i, bg_sub_module in enumerate(bg_sub_modules):
            setattr(self, 'bg_sub_module_{}'.format(i), bg_sub_module)

        for i, image_embedding in enumerate(image_embeddings):
            setattr(self, 'image_embedding_{}'.format(i), image_embedding)

        self.centroids = centroids
        self.grid_dim = grid_dim
        self.min_position = min_position
        self.max_position = max_position
        self.need_viewdir = need_viewdir
        self.need_appearance_embedding = len(image_embeddings) > 0
