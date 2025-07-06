import torch, torch.nn as nn, torch.nn.functional as F, torch_geometric as pyg

from encoder import text_encoder, image_encoder, audio_encoder

# MIG = Multimodal Intermediate Graph Representations
class MIGR(nn.Module):
    def __init__(self, modalities:list[str] = ['text', 'image']):
        super(MIGR, self).__init__()

        # TODO: model init
        self.encoders = nn.ModuleDict()
    
        if 'text' in modalities:
            self.encoders['text'] = text_encoder()
        if 'image' in modalities:
            self.encoders['image'] = image_encoder()
        if 'audio' in modalities:
            self.encoders['audio'] = audio_encoder()

        # GNN
        self.GNN = pyg.nn.conv.GPSConv()

        # Pred head
        self.pred_head = ...

    def forward(self, x):
        # TODO: model forward()
        return x