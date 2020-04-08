''' Define the Layers '''
import torch.nn as nn
from Models.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Fenglin Liu"
# Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations. NeurIPS 2019. #

class RefineVisualLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(RefineVisualLayer, self).__init__()
        self.enc_attn = MultiHeadAttention( n_head, d_model, d_k, d_v, dropout=dropout )
        self.pos_ffn = PositionwiseFeedForward( d_model, d_inner, dropout=dropout )
        self.layer_norm = nn.LayerNorm( d_model )

    def forward(self, T, V):
        residual = V
        Refine_V = self.enc_attn( T, V, V )

        Refine_V = self.pos_ffn( Refine_V )
        Refine_V = self.layer_norm( Refine_V + residual )

        return Refine_V

class RefineTextualLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(RefineTextualLayer, self).__init__()
        self.enc_attn = MultiHeadAttention( n_head, d_model, d_k, d_v, dropout=dropout )
        self.pos_ffn = PositionwiseFeedForward( d_model, d_inner, dropout=dropout )

    def forward(self, V, T):

        Refine_T = self.enc_attn( V, T, T )

        Refine_T = self.pos_ffn( Refine_T )

        return Refine_T
