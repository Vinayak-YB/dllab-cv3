import os
import json

from einops import rearrange
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, norm_type=None, norm_groups=None):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # Normalization applied to the output hidden state h
        if norm_type == 'group':
            if norm_groups is None:
                raise ValueError("norm_groups must be specified for group normalization")
            # Ensure norm_groups divides the number of channels
            if hidden_channels % norm_groups != 0:
                 raise ValueError(f"norm_groups ({norm_groups}) must divide the hidden_channels ({hidden_channels})")
            self.norm = nn.GroupNorm(norm_groups, hidden_channels)
        elif norm_type == 'layer':
            # LayerNorm equivalent using GroupNorm with 1 group
            self.norm = nn.GroupNorm(1, hidden_channels)
        elif norm_type is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Choose 'group', 'layer', or None.")

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)  # (B, C+H, H, W)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = self.norm(h_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels_list, kernel_size=3, norm_type=None, norm_groups=None):
        super().__init__()
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = len(hidden_channels_list)

        if norm_type == 'group' and norm_groups is None:
            raise ValueError("norm_groups must be specified for group normalization in ConvLSTM")

        layers = []
        for i in range(self.num_layers):
            in_channels = input_channels if i == 0 else hidden_channels_list[i - 1]
            # Pass norm parameters to the cell
            cell = ConvLSTMCell(in_channels, hidden_channels_list[i], kernel_size, norm_type, norm_groups)
            layers.append(cell)

        self.layers = nn.ModuleList(layers)

    def forward(self, input_seq, h_list=None, c_list=None):
        # input_seq: (B, T, C, H, W)
        B, T, _, H, W = input_seq.size()
        device = input_seq.device

        if h_list is None or c_list is None:
            h_list, c_list = self.init_hidden(B, H, W, device)

        outputs = [] # Store hidden states for each time step if needed, currently only last is used
        for t in range(T):
            x = input_seq[:, t]  # (B, C, H, W)
            for l, cell in enumerate(self.layers):
                h_list[l], c_list[l] = cell(x, h_list[l], c_list[l])
                x = h_list[l]  # output of this layer is input to next
            outputs.append(h_list[-1]) # Append last layer's hidden state for this time step

        # Return only the last hidden state of the last layer
        return h_list[-1]

    def init_hidden(self, batch_size, height, width, device):
        h_list = []
        c_list = []
        for i in range(self.num_layers):
            h, c = self.layers[i].init_hidden(batch_size, height, width, device)
            h_list.append(h)
            c_list.append(c)
        return h_list, c_list


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, norm_type=None, norm_groups=None):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # Gates: update (z) and reset (r)
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=padding
        )

        # Candidate hidden state
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding
        )

        # Normalization applied to the output hidden state h
        if norm_type == 'group':
            if norm_groups is None:
                raise ValueError("norm_groups must be specified for group normalization")
            # Ensure norm_groups divides the number of channels
            if hidden_channels % norm_groups != 0:
                 raise ValueError(f"norm_groups ({norm_groups}) must divide the hidden_channels ({hidden_channels})")
            self.norm = nn.GroupNorm(norm_groups, hidden_channels)
        elif norm_type == 'layer':
            # LayerNorm equivalent using GroupNorm with 1 group
            self.norm = nn.GroupNorm(1, hidden_channels)
        elif norm_type is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Choose 'group', 'layer', or None.")


    def forward(self, x, h):
        # x: (B, C, H, W), h: (B, H_hidden, H, W)
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        z, r = torch.chunk(gates, 2, dim=1)

        z = torch.sigmoid(z)  # Update gate
        r = torch.sigmoid(r)  # Reset gate

        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde_raw = self.candidate(combined_reset)
        h_tilde = torch.tanh(h_tilde_raw)

        # Calculate next hidden state
        h_next= (1 - z) * h + z * h_tilde
        h_next = self.norm(h_next)

        return h_next

    def init_hidden(self, batch_size, height, width, device):
        return torch.zeros(batch_size, self.hidden_channels, height, width, device=device)


class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels_list, kernel_size=3, norm_type=None, norm_groups=None):
        super().__init__()
        self.hidden_channels_list = hidden_channels_list
        self.num_layers = len(hidden_channels_list)

        if norm_type == 'group' and norm_groups is None:
            raise ValueError("norm_groups must be specified for group normalization in ConvGRU")

        layers = []
        for i in range(self.num_layers):
            in_channels = input_channels if i == 0 else hidden_channels_list[i - 1]
            # Pass norm parameters to the cell
            cell = ConvGRUCell(in_channels, hidden_channels_list[i], kernel_size, norm_type, norm_groups)
            layers.append(cell)

        self.layers = nn.ModuleList(layers)

    def forward(self, input_seq, h_list=None):
        # input_seq: (B, T, C, H, W)
        B, T, _, H, W = input_seq.size()
        device = input_seq.device

        if h_list is None:
            h_list = self.init_hidden(B, H, W, device)

        outputs = []
        for t in range(T):
            x = input_seq[:, t]  # (B, C, H, W)
            for l, cell in enumerate(self.layers):
                h_list[l] = cell(x, h_list[l])
                x = h_list[l]  # output of this layer is input to next
                if self.training:
                    x += torch.randn_like(x) * 0.02  # Add noise during training
            outputs.append(h_list[-1])

        # Return only the last hidden state of the last layer
        return h_list[-1]

    def init_hidden(self, batch_size, height, width, device):
        h_list = []
        for i in range(self.num_layers):
            h = self.layers[i].init_hidden(batch_size, height, width, device)
            h_list.append(h)
        return h_list


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_channels, n_layers=1, d_model=128, n_heads=4, patch_size=16, img_size=128, context=5):
        super().__init__()

        self.patch_embed = nn.Conv2d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos = nn.Parameter(0.02 * torch.randn(1, 1, (img_size // patch_size) ** 2, d_model))
        self.temporal_pos = nn.Parameter(0.02 * torch.randn(1, context, 1, d_model))

        self.n_layers = n_layers
        self.spatial_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4) for _ in range(n_layers)]
        )
        self.temporal_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // patch_size, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(),
            nn.Conv2d(d_model // patch_size, input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # create patch embeddings per frame
        B, T, _, H, W = x.shape
        x = rearrange(x, 'B T C H W -> (B T) C H W')  # (B * T, C, H, W)
        x = self.patch_embed(x)  # (B * T, d_model, H/patch_size, W/patch_size)

        # add spatial and temporal positional encodings
        x = rearrange(x, '(B T) D H W -> B T (H W) D', B=B, T=T)  # (B, T, N, d_model)
        x = x + self.spatial_pos + self.temporal_pos

        N = x.shape[2]  # Number of patches
        x = rearrange(x, 'B T N D -> (B T) N D')  # (B * T, N, d_model)
        for i in range(self.n_layers):
            x = self.spatial_blocks[i](x)
            x = rearrange(x, '(B T) N D -> (B N) T D', B=B, T=T)  # (B * N, T, d_model)

            x = self.temporal_blocks[i](x)
            x = rearrange(x, '(B N) T D -> (B T) N D', B=B, N=N)  # (B, T, N * d_model)

        x = rearrange(x, '(B T) N D -> B T N D', B=B, T=T)  # (B, T, N, d_model)
        x = x[:, -1]  # Get the last time step
        x = self.norm(x)

        H, W = int(H / self.patch_embed.stride[0]), int(W / self.patch_embed.stride[1])
        x = rearrange(x, 'B (H W) D -> B D H W', H=H, W=W)
        x = self.decoder(x)

        return x


class SequenceEncoderDecoder(nn.Module):
    def __init__(self, rnn_type='lstm', input_channels=1, hidden_channels=[32], kernel_size=3, encoder_channels=16, **kwargs):
        super().__init__()
        if not hidden_channels:
            raise ValueError('hidden_channels_list cannot be empty')

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        rnn_input_channels = encoder_channels
        if rnn_type.lower() == 'lstm':
            self.rnn = ConvLSTM(rnn_input_channels, hidden_channels, kernel_size, **kwargs)
        elif rnn_type.lower() == 'gru':
            self.rnn = ConvGRU(rnn_input_channels, hidden_channels, kernel_size, **kwargs)
        else:
            raise ValueError(f'Unsupported RNN type: {rnn_type}. Choose \'lstm\' or \'gru\'.')

        rnn_output_channels = hidden_channels[-1]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(rnn_output_channels, rnn_output_channels // 2, kernel_size=2, stride=2),
            nn.SiLU(),
            nn.Conv2d(rnn_output_channels // 2, input_channels, kernel_size=3, padding=1),
        )

    def forward(self, input_seq, return_last_hidden_state=False):
        # input_seq: (B, T, C, H, W)
        B, T, C, H, W = input_seq.size()

        # Encode
        encoded_seq = self.encoder(input_seq.view(B * T, C, H, W))  # (B*T, EncC, H/2, W/2)
        encoded_seq = encoded_seq.view(B, T, *encoded_seq.shape[1:])  # (B, T, EncC, H/2, W/2)

        # RNN processing
        last_hidden_state = self.rnn(encoded_seq) # Get last hidden state from last layer

        # Decode
        out = self.decoder(last_hidden_state)  # Upsample to original resolution (B, C, H, W)
        return (out, last_hidden_state) if return_last_hidden_state else out

    
    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str,
        ckpt_name: str = "last.ckpt",
        device: str = "cpu",
    ):
        with open(os.path.join(ckpt_dir, "args.json")) as f:
            args = json.load(f)

        # ----------------------------
        # Construct the correct model
        # ----------------------------
        if args["rnn_type"].lower() == "stt":
            model = SpatioTemporalTransformer(
                input_channels=args["input_channels"],
                n_layers=args["transformer_layers"],
                d_model=args["transformer_dim"],
                n_heads=args["transformer_heads"],
                patch_size=args["patch_size"],
                img_size=args["image_size"],
                context=args["context"],
            )
        else:
            model = cls(
                rnn_type=args["rnn_type"],
                input_channels=args["input_channels"],
                hidden_channels=args["hidden_channels"],
                kernel_size=args["kernel_size"],
                encoder_channels=args["encoder_channels"],
                norm_type=args["norm_type"],
                norm_groups=args["norm_groups"],
            )

        model.to(device)

        # ----------------------------
        # Load checkpoint
        # ----------------------------
        state_dict = torch.load(
            os.path.join(ckpt_dir, "checkpoints", ckpt_name),
            map_location=device,
            weights_only=True,
        )["state_dict"]

        state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        return model


if __name__ == '__main__':
    model = SpatioTemporalTransformer(
        input_channels=1,
        n_layers=2,
        d_model=64,
        n_heads=4,
        patch_size=16,
        img_size=128,
        context=5
    )
    input_tensor = torch.randn(2, 5, 1, 128, 128)  # (B, T, C, H, W)
    output = model(input_tensor)
    print('Output shape:', output.shape)  # Should be (B, C, H, W)

    print(sum(p.numel() for p in model.spatial_blocks.parameters()))  # Print number of trainable parameters
    # print(set([n for n, p in model.named_parameters()]))
