import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="group", groups=8):
        super().__init__()

        def make_norm(c):
            if norm == "group":
                g = min(groups, c)
                while c % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, c)
            elif norm == "batch":
                return nn.BatchNorm2d(c)
            elif norm == "layer":
                return nn.GroupNorm(1, c)
            else:
                return nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            make_norm(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class FrameEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=32, latent_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels * 2),

            nn.Conv2d(base_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(latent_channels),
            
            # Normalizzazione essenziale per il Flow Matching: 
            # compatibilità con rumore Gaussiano N(0, 1)
            nn.GroupNorm(8, latent_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class FrameDecoder(nn.Module):
    def __init__(self, latent_channels=64, base_channels=32, output_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(latent_channels),

            nn.ConvTranspose2d(latent_channels, base_channels * 2, kernel_size=2, stride=2),
            nn.SiLU(),
            ResidualBlock(base_channels * 2),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.SiLU(),
            ResidualBlock(base_channels),

            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, spatial_size, device):
        h, w = spatial_size
        h0 = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        c0 = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return h0, c0


class LatentDynamicsConvLSTM(nn.Module):
    def __init__(self, latent_channels=64, hidden_channels=64, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(latent_channels, hidden_channels, kernel_size=kernel_size)
        self.proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)

    def forward(self, z_seq):
        """
        z_seq: (B, T, C, H, W)
        returns:
            dyn_context: (B, C, H, W)
            last_hidden: (B, Hc, H, W)
        """
        b, t, c, h, w = z_seq.shape
        h_t, c_t = self.cell.init_state(b, (h, w), z_seq.device)

        for i in range(t):
            h_t, c_t = self.cell(z_seq[:, i], h_t, c_t)

        dyn_context = self.proj(h_t)
        return dyn_context, h_t


class ContextAggregator(nn.Module):
    def __init__(self, latent_channels=64, context_frames=5):
        super().__init__()
        self.context_frames = context_frames
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels * context_frames, latent_channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(latent_channels * 2),
            nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, z_seq):
        b, t, c, h, w = z_seq.shape
        if t != self.context_frames:
            raise ValueError(f"Expected {self.context_frames} context frames, got {t}")
        x = z_seq.reshape(b, t * c, h, w)
        return self.net(x)


class StateHead(nn.Module):
    """
    Predicts explicit physical state (x, y, vx, vy) from latent context.
    """
    def __init__(self, latent_channels=64, hidden_dim=128, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(latent_channels * 4 * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class FlowMatchingNet(nn.Module):
    def __init__(self, latent_channels=64, time_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, latent_channels),
            nn.SiLU(),
            nn.Linear(latent_channels, latent_channels),
        )

        self.net = nn.Sequential(
            nn.Conv2d(latent_channels * 3, latent_channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(latent_channels * 2),
            nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_t, t, context_static, context_dynamic):
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(-1).unsqueeze(-1)
        z_in = z_t + t_emb
        x = torch.cat([z_in, context_static, context_dynamic], dim=1)
        return self.net(x)


class LatentFlowVideoPredictor(nn.Module):
    def __init__(
        self,
        input_channels=1,
        base_channels=32,
        latent_channels=64,
        context_frames=5,
        time_dim=64,
        dynamics_hidden_channels=64,
        state_loss_weight=0.1,
        recon_loss_weight=0.2,
        motion_loss_weight=0.1,
        invert=False,
        generated_frame_loss_weight=0.2,
        generation_loss_steps=5,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.context_frames = context_frames

        self.state_loss_weight = state_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.motion_loss_weight = motion_loss_weight
        self.generated_frame_loss_weight = generated_frame_loss_weight
        self.generation_loss_steps = generation_loss_steps

        self.encoder = FrameEncoder(
            input_channels=input_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        self.decoder = FrameDecoder(
            latent_channels=latent_channels,
            base_channels=base_channels,
            output_channels=input_channels,
        )

        self.context_net = ContextAggregator(
            latent_channels=latent_channels,
            context_frames=context_frames,
        )

        self.dynamics_net = LatentDynamicsConvLSTM(
            latent_channels=latent_channels,
            hidden_channels=dynamics_hidden_channels,
            kernel_size=3,
        )

        self.state_head = StateHead(
            latent_channels=latent_channels,
            hidden_dim=128,
            out_dim=4,
        )

        self.flow_net = FlowMatchingNet(
            latent_channels=latent_channels,
            time_dim=time_dim,
        )

        self.invert = invert

    def encode_sequence(self, input_seq):
        b, t, c, h, w = input_seq.shape
        x = input_seq.reshape(b * t, c, h, w)
        z = self.encoder(x)
        _, c_lat, h_lat, w_lat = z.shape
        z = z.reshape(b, t, c_lat, h_lat, w_lat)
        return z

    def encode_frame(self, frame):
        return self.encoder(frame)

    def decode_latent(self, z):
        return self.decoder(z)

    def get_context_representation(self, context_frames):
        z_ctx = self.encode_sequence(context_frames)
        context_static = self.context_net(z_ctx)
        context_dynamic, dyn_hidden = self.dynamics_net(z_ctx)
        context = context_static + context_dynamic
        return {
            "z_ctx": z_ctx,
            "context_static": context_static,
            "context_dynamic": context_dynamic,
            "context": context,
            "dyn_hidden": dyn_hidden,
        }

    def predict_state(self, context_frames):
        rep = self.get_context_representation(context_frames)
        state_pred = self.state_head(rep["context"])
        return state_pred

    def compute_loss(self, context_frames, target_frame, target_state=None):
        """
        target_state: optional tensor (B, 4) = [x, y, vx, vy]
        """
        rep = self.get_context_representation(context_frames)
        context_static = rep["context_static"]
        context_dynamic = rep["context_dynamic"]
        context = rep["context"]

        z_target = self.encode_frame(target_frame)
        noise = torch.randn_like(z_target)

        t = torch.rand(z_target.size(0), device=z_target.device)
        t_view = t[:, None, None, None]

        z_t = (1.0 - t_view) * noise + t_view * z_target
        v_target = z_target - noise

        # 1. Flow Loss: Addestriamo la rete sul campo vettoriale
        v_pred = self.flow_net(z_t, t, context_static, context_dynamic)
        flow_loss = F.mse_loss(v_pred, v_target)

        # Extra generated-frame loss:
        # train the actual inference path noise -> flow -> generated latent -> decoder.
        # This helps avoid the mismatch where the decoder only sees clean z_target
        # during training but receives flow-generated latents during prediction.
        gen_steps = max(1, int(self.generation_loss_steps))
        z_gen = noise
        dt_gen = 1.0 / gen_steps

        for i in range(gen_steps):
            t_gen = torch.full(
                (z_target.size(0),),
                i / gen_steps,
                device=z_target.device,
            )
            v_gen = self.flow_net(z_gen, t_gen, context_static, context_dynamic)
            z_gen = z_gen + dt_gen * v_gen

        pred_frame_from_flow = self.decode_latent(z_gen)

        # 2. Recon Loss e Motion Loss: Addestriamo il Decoder dal Latente Pulito
        recon_clean = self.decode_latent(z_target)
        
        fg_weight = 15.0
        weights = torch.ones_like(target_frame)

        if self.invert:
            # after invert: ball is bright, background is dark
            foreground_mask = target_frame > 0.5
        else:
            # without invert: ball is dark, background is bright
            foreground_mask = target_frame < 0.5

        weights[foreground_mask] = fg_weight
        
        recon_loss = (weights * (recon_clean - target_frame) ** 2).mean()

        generated_frame_loss = (weights * (pred_frame_from_flow - target_frame) ** 2).mean()

        last_context = context_frames[:, -1]
        pred_motion = recon_clean - last_context
        tgt_motion = target_frame - last_context
        motion_loss = (weights * (pred_motion - tgt_motion) ** 2).mean()

        # 3. State Loss (PINN constraint)
        if target_state is not None:
            state_pred = self.state_head(context)
            state_loss = F.mse_loss(state_pred, target_state)
        else:
            state_pred = None
            state_loss = torch.tensor(0.0, device=target_frame.device)

        total_loss = (
            flow_loss
            + self.recon_loss_weight * recon_loss
            + self.motion_loss_weight * motion_loss
            + self.state_loss_weight * state_loss
            + self.generated_frame_loss_weight * generated_frame_loss
        )

        return {
            "loss": total_loss,
            "flow_loss": flow_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "motion_loss": motion_loss.detach(),
            "state_loss": state_loss.detach(),
            "generated_frame_loss": generated_frame_loss.detach(),
            "recon_frame": recon_clean.detach(),
            "generated_frame": pred_frame_from_flow.detach(),
            "state_pred": state_pred.detach() if state_pred is not None else None,
        }

    @torch.no_grad()
    def predict_next_frame(self, context_frames, num_steps=20, return_latent=False):
        self.eval()

        rep = self.get_context_representation(context_frames)
        context_static = rep["context_static"]
        context_dynamic = rep["context_dynamic"]

        b, _, c_lat, h_lat, w_lat = rep["z_ctx"].shape
        z = torch.randn(b, c_lat, h_lat, w_lat, device=context_frames.device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), i / num_steps, device=context_frames.device)
            v = self.flow_net(z, t, context_static, context_dynamic)
            z = z + dt * v

        next_frame = self.decode_latent(z)

        if return_latent:
            return next_frame, z
        return next_frame

    @torch.no_grad()
    def rollout(self, context_frames, pred_steps, num_steps=20):
        self.eval()

        preds = []
        current_context = context_frames.clone()

        for _ in range(pred_steps):
            next_frame = self.predict_next_frame(current_context, num_steps=num_steps)
            preds.append(next_frame)
            current_context = torch.cat(
                [current_context[:, 1:], next_frame.unsqueeze(1)],
                dim=1
            )

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    model = LatentFlowVideoPredictor(
        input_channels=1,
        base_channels=32,
        latent_channels=64,
        context_frames=5,
        time_dim=64,
        dynamics_hidden_channels=64,
        state_loss_weight=0.1,
        recon_loss_weight=0.2,
        motion_loss_weight=0.1,
    )

    x = torch.randn(2, 5, 1, 128, 128)
    y = torch.randn(2, 1, 128, 128)
    s = torch.randn(2, 4)

    losses = model.compute_loss(x, y, target_state=s)
    print("Total loss:", losses["loss"].item())
    print("Flow loss:", losses["flow_loss"].item())
    print("Recon loss:", losses["recon_loss"].item())
    print("Motion loss:", losses["motion_loss"].item())
    print("State loss:", losses["state_loss"].item())

    pred = model.predict_next_frame(x, num_steps=20)
    print("Pred next frame shape:", pred.shape)

    rollout = model.rollout(x, pred_steps=10, num_steps=20)
    print("Rollout shape:", rollout.shape)

    state_pred = model.predict_state(x)
    print("State pred shape:", state_pred.shape)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)