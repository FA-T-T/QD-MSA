"""Implements common unimodal encoders."""
import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRU(torch.nn.Module):
    """Implements Gated Recurrent Unit (GRU)."""
    
    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, last_only=False,batch_first=True):
        """Initialize GRU Module.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden dimension
            dropout (bool, optional): Whether to apply dropout layer. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output. Defaults to False.
            has_padding (bool, optional): Whether input has padding. Defaults to False.
            last_only (bool, optional): Whether to return only the last output. Defaults to False.
            batch_first (bool, optional): Whether batch dimension is first. Defaults to True.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        """Apply GRU to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]
            
            
            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)
        
        return out


class GRUWithLinear(torch.nn.Module):
    """Implements a GRU with Linear Post-Processing."""
    
    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, output_each_layer=False, batch_first=False):
        """Initialize GRUWithLinear Module.

        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Dimension
            outdim (int): Output Dimension
            dropout (bool, optional): Whether to apply dropout. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output. Defaults to False.
            has_padding (bool, optional): Whether input has padding. Defaults to False.
            output_each_layer (bool, optional): Return output of each intermediate layer. Defaults to False.
            batch_first (bool, optional): Batch dimension first in GRU. Defaults to False.
        """
        super(GRUWithLinear, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=batch_first)
        self.linear = nn.Linear(hiddim, outdim)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply GRUWithLinear to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        out = self.linear(hidden)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.output_each_layer:
            return [0, torch.flatten(x, 1), torch.flatten(hidden, 1), self.lklu(out)]
        return out

class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self, masks=None):
        """Initialize Concat Module."""
        super(Concat, self).__init__()
        self.masks = masks

    def forward(self, modalities):
        """
        Forward Pass of Concat.

        :param modalities: An iterable of modalities to combine
        :param masks: Optional indices to select specific modalities
        """
        if self.masks is None:
            masks = range(len(modalities))  # Use all modalities by default
        
        flattened = []
        for idx in self.masks:
            modality = modalities[idx]
            flattened.append(torch.flatten(modality, start_dim=1))
        
        return torch.cat(flattened, dim=1)
