"""Version-controlled extensions for dynamically loaded InternVL models."""

from .gosa import GlobalTilePositionalEncoding, Learnable2DPositionalEncoding, attach_gosa
from .goar import GOSAOCRAnchorRefinement, attach_goar, find_goar_host, get_active_goar_adapter

__all__ = [
    'GlobalTilePositionalEncoding', 'Learnable2DPositionalEncoding', 'GOSAOCRAnchorRefinement',
    'attach_gosa', 'attach_goar', 'find_goar_host', 'get_active_goar_adapter'
]
