import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, norm_type=None, norm_groups=None):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.candidate = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        if norm_type == "group":
            if norm_groups is None:
                raise ValueError("norm_groups must be specified when norm_type='group'")
            if hidden_channels % norm_groups != 0:
                raise ValueError(f"norm_groups ({norm_groups}) must divide hidden_channels ({hidden_channels})")
            self.norm = nn.GroupNorm(norm_groups, hidden_channels)
        elif norm_type == "layer":
            self.norm = nn.GroupNorm(1, hidden_channels)
        elif norm_type is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        z, r = torch.chunk(gates, 2, dim=1)

        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.candidate(combined_reset))

        h_next = (1 - z) * h + z * h_tilde
        h_next = self.norm(h_next)
        return h_next

    def init_hidden(self, batch_size, height, width, device):
        return torch.zeros(batch_size, self.hidden_channels, height, width, device=device)


class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels_list, kernel_size=3, norm_type=None, norm_groups=None):
        super().__init__()
        if not hidden_channels_list:
            raise ValueError("hidden_channels_list cannot be empty")

        self.hidden_channels_list = hidden_channels_list
        self.num_layers = len(hidden_channels_list)

        layers = []
        for i in range(self.num_layers):
            in_channels = input_channels if i == 0 else hidden_channels_list[i - 1]
            layers.append(
                ConvGRUCell(
                    input_channels=in_channels,
                    hidden_channels=hidden_channels_list[i],
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                    norm_groups=norm_groups
                )
            )
        self.layers = nn.ModuleList(layers)

    def init_hidden(self, batch_size, height, width, device):
        return [
            layer.init_hidden(batch_size, height, width, device)
            for layer in self.layers
        ]

    def forward(self, input_seq, h_list=None):
        """
        input_seq: (B, T, C, H, W)
        returns:
            outputs: list of hidden states from the last layer for each time step, len T
            h_list: final hidden states for all layers
        """
        B, T, _, H, W = input_seq.shape
        device = input_seq.device

        if h_list is None:
            h_list = self.init_hidden(B, H, W, device)

        outputs = []
        for t in range(T):
            x = input_seq[:, t]
            for l, cell in enumerate(self.layers):
                h_list[l] = cell(x, h_list[l])
                x = h_list[l]
            outputs.append(h_list[-1])

        return outputs, h_list


class FrameEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class FrameDecoder(nn.Module):
    def __init__(self, hidden_channels, output_channels=1):
        super().__init__()
        mid_channels = max(hidden_channels // 2, 16)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, mid_channels, kernel_size=2, stride=2),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, kernel_size=2, stride=2),
            nn.SiLU(),
            nn.Conv2d(mid_channels // 2, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class BaselineVideoPredictor(nn.Module):
    """
    Baseline recurrent video predictor:
    - Encode each frame with CNN
    - Process sequence with ConvGRU
    - Decode last hidden state to predict next frame
    - Supports autoregressive rollout
    """
    def __init__(
        self,
        input_channels=1,
        encoder_channels=32,
        hidden_channels=[64],
        kernel_size=3,
        norm_type=None,
        norm_groups=None
    ):
        super().__init__()

        self.input_channels = input_channels
        self.encoder_channels = encoder_channels
        self.hidden_channels = hidden_channels

        self.encoder = FrameEncoder(
            input_channels=input_channels,
            base_channels=encoder_channels
        )

        self.rnn = ConvGRU(
            input_channels=encoder_channels,
            hidden_channels_list=hidden_channels,
            kernel_size=kernel_size,
            norm_type=norm_type,
            norm_groups=norm_groups
        )

        self.decoder = FrameDecoder(
            hidden_channels=hidden_channels[-1],
            output_channels=input_channels
        )

    def encode_sequence(self, input_seq):
        """
        input_seq: (B, T, C, H, W)
        returns encoded_seq: (B, T, C_enc, H_enc, W_enc)
        """
        B, T, C, H, W = input_seq.shape
        x = input_seq.reshape(B * T, C, H, W)
        x = self.encoder(x)
        _, C_enc, H_enc, W_enc = x.shape
        x = x.reshape(B, T, C_enc, H_enc, W_enc)
        return x

    def forward(self, input_seq, return_hidden=False):
        """
        input_seq: (B, T, C, H, W)
        returns next_frame: (B, C, H, W)
        """
        encoded_seq = self.encode_sequence(input_seq)
        outputs, h_list = self.rnn(encoded_seq)
        last_hidden = outputs[-1]
        next_frame = self.decoder(last_hidden)

        if return_hidden:
            return next_frame, h_list
        return next_frame

    @torch.no_grad()
    def rollout(self, context_frames, pred_steps):
        """
        Autoregressive rollout.

        context_frames: (B, T_ctx, C, H, W)
        pred_steps: int

        returns:
            pred_seq: (B, pred_steps, C, H, W)
        """
        self.eval()

        preds = []
        current_context = context_frames.clone()

        for _ in range(pred_steps):
            next_frame = self.forward(current_context)
            preds.append(next_frame)
            current_context = torch.cat(
                [current_context[:, 1:], next_frame.unsqueeze(1)],
                dim=1
            )

        return torch.stack(preds, dim=1)

    def rollout_with_hidden(self, context_frames, pred_steps):
        """
        Slightly more efficient rollout:
        - encode context once
        - keep hidden state
        - feed predictions one by one

        returns:
            pred_seq: (B, pred_steps, C, H, W)
        """
        B, T, C, H, W = context_frames.shape
        device = context_frames.device

        encoded_context = self.encode_sequence(context_frames)
        _, h_list = self.rnn(encoded_context)

        last_frame = context_frames[:, -1]
        preds = []

        for _ in range(pred_steps):
            encoded_last = self.encoder(last_frame)
            outputs, h_list = self.rnn(encoded_last.unsqueeze(1), h_list=h_list)
            next_hidden = outputs[-1]
            next_frame = self.decoder(next_hidden)

            preds.append(next_frame)
            last_frame = next_frame

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    model = BaselineVideoPredictor(
        input_channels=1,
        encoder_channels=32,
        hidden_channels=[64],
        kernel_size=3,
        norm_type=None
    )

    x = torch.randn(2, 5, 1, 128, 128)  # (B, T, C, H, W)

    y = model(x)
    print("Next frame shape:", y.shape)  # (2, 1, 128, 128)

    rollout = model.rollout(x, pred_steps=10)
    print("Rollout shape:", rollout.shape)  # (2, 10, 1, 128, 128)

    rollout_fast = model.rollout_with_hidden(x, pred_steps=10)
    print("Rollout with hidden shape:", rollout_fast.shape)  # (2, 10, 1, 128, 128)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)