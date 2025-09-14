import torch
import torch.nn as nn

class HitExciter(nn.Module):
    def __init__(self, T, dt, node_ids):
        super().__init__()
        self.t0   = nn.Parameter(torch.tensor(0.05))   # seconds
        self.sigma= nn.Parameter(torch.tensor(0.003))  # s, width
        self.A    = nn.Parameter(torch.tensor(5.0))    # force amplitude
        self.dirv = nn.Parameter(torch.tensor([1.0,0.0,0.0]))  # 3D, learnable
        self.dt, self.T, self.node_ids = dt, T, node_ids
    def forward(self, t_idx, pos, vel):
        t  = t_idx * self.dt
        env = torch.exp(-0.5 * ((t - torch.sigmoid(self.t0)*self.T*self.dt) /
                                (self.sigma.abs() + 1e-6))**2)
        d  = self.dirv / (self.dirv.norm()+1e-9)
        f  = self.A * env * d   # (3,)
        return self.node_ids, f

class PluckExciter(nn.Module):
    def __init__(self, dt, node_ids):
        super().__init__()
        self.A     = nn.Parameter(torch.tensor(0.01))  # virtual pull distance
        self.kp    = nn.Parameter(torch.tensor(200.0)) # spring to pick point
        self.t_rel = nn.Parameter(torch.tensor(0.04))  # release time
        self.dirv  = nn.Parameter(torch.tensor([1.,0.,0.]))
        self.dt, self.node_ids = dt, node_ids
    def forward(self, t_idx, pos, vel):
        t   = t_idx * self.dt
        d   = self.dirv / (self.dirv.norm()+1e-9)
        # virtual pick point trajectory: ramp to A then release
        hold = torch.sigmoid( (self.t_rel - t)/ (1e-3) )  # ~1 before release, 0 after
        x_target = hold * (self.A * d)                    # displacement target
        # Hooke force to target (on each driver node): F = kp * (x_target - x)
        i = self.node_ids
        x = pos[i].mean(dim=0)   # simple avg if multiple drivers
        f = self.kp.abs() * (x_target - x)
        return self.node_ids, f
    
class BowExciter(nn.Module):
    def __init__(self, dt, node_id):
        super().__init__()
        self.v_bow = nn.Parameter(torch.tensor(0.3))   # tangential m/s (scaled)
        self.N     = nn.Parameter(torch.tensor(2.0))   # normal force
        self.mu_s  = nn.Parameter(torch.tensor(0.8))
        self.mu_c  = nn.Parameter(torch.tensor(0.4))
        self.v_s   = nn.Parameter(torch.tensor(0.05))
        self.t_hat = nn.Parameter(torch.tensor([1.,0.,0.])) # along string
        self.dt, self.node = dt, node_id
    def forward(self, t_idx, pos, vel):
        d = self.t_hat / (self.t_hat.norm()+1e-9)
        i = self.node
        v_string = (pos[i] - vel[i]*0 + (pos[i]-pos[i]))  # placeholder for IDE
        v_string = (pos[i] - pos[i])                      # (ignored, see below)
        v_i = vel[i].dot(d)                                # tangential velocity
        v_rel = v_i - self.v_bow
        mu = self.mu_c + (self.mu_s - self.mu_c) * torch.exp(-(v_rel/(self.v_s.abs()+1e-6))**2)
        f = - self.N.abs() * mu * torch.tanh(v_rel / 1e-3) * d
        return torch.tensor([self.node], device=pos.device), f

    def __init__(self, dt, node_ids, d_hat=(1,0,0), B=8):
        super().__init__()
        self.dt = dt
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        self.d_hat = nn.Parameter(torch.tensor(d_hat, dtype=torch.float32))
        # simple RBF envelope for mouth pressure
        self.T = None  # set at runtime if you want full vector
        centers = torch.linspace(0, 1, B)
        widths  = torch.full((B,), 0.12)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.theta_P = nn.Parameter(torch.zeros(B))        # Pm coeffs
        self.theta_Z = nn.Parameter(torch.tensor(0.2))     # >0
        self.theta_G = nn.Parameter(torch.tensor(2.0))     # >0

    def _env(self, t_norm):
        # t_norm in [0,1]
        phi = torch.exp(-0.5*((t_norm[:,None]-self.centers[None,:])/
                              (self.widths[None,:]+1e-6))**2)  # [T,B]
        w = self.theta_P                                    # signed
        return (phi @ w).squeeze(-1)                        # [T]

    def forward(self, t_idx, pos, vel, T_total):
        # time-normalized envelope value
        t_norm = torch.linspace(0, 1, T_total, device=pos.device, dtype=pos.dtype)
        Pm_t = self._env(t_norm)[t_idx]                      # scalar
        Z = torch.nn.functional.softplus(self.theta_Z) + 1e-6
        G = torch.nn.functional.softplus(self.theta_G)
        d = self.d_hat / (self.d_hat.norm()+1e-9)

        # project node velocity along d and average across driver nodes
        i = self.node_ids.to(pos.device)
        v_par = vel[i].mean(0).dot(d)                        # scalar
        F_scalar = G * torch.tanh(Pm_t - Z * v_par)
        F_vec = F_scalar * d                                 # (3,)
        return i, F_vec

class BlowExciter(nn.Module):
    def __init__(self, dt, node_ids, d_hat=(1,0,0), B=8):
        super().__init__()
        self.dt = dt
        self.node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        self.d_hat = nn.Parameter(torch.tensor(d_hat, dtype=torch.float32))
        # simple RBF envelope for mouth pressure
        self.T = None  # set at runtime if you want full vector
        centers = torch.linspace(0, 1, B)
        widths  = torch.full((B,), 0.12)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.theta_P = nn.Parameter(torch.zeros(B))        # Pm coeffs
        self.theta_Z = nn.Parameter(torch.tensor(0.2))     # >0
        self.theta_G = nn.Parameter(torch.tensor(2.0))     # >0

    def _env(self, t_norm):
        # t_norm in [0,1]
        phi = torch.exp(-0.5*((t_norm[:,None]-self.centers[None,:])/
                              (self.widths[None,:]+1e-6))**2)  # [T,B]
        w = self.theta_P                                    # signed
        return (phi @ w).squeeze(-1)                        # [T]

    def forward(self, t_idx, pos, vel, T_total):
        # time-normalized envelope value
        t_norm = torch.linspace(0, 1, T_total, device=pos.device, dtype=pos.dtype)
        Pm_t = self._env(t_norm)[t_idx]                      # scalar
        Z = torch.nn.functional.softplus(self.theta_Z) + 1e-6
        G = torch.nn.functional.softplus(self.theta_G)
        d = self.d_hat / (self.d_hat.norm()+1e-9)

        # project node velocity along d and average across driver nodes
        i = self.node_ids.to(pos.device)
        v_par = vel[i].mean(0).dot(d)                        # scalar
        F_scalar = G * torch.tanh(Pm_t - Z * v_par)
        F_vec = F_scalar * d                                 # (3,)
        return i, F_vec


class ExciterBank(nn.Module):
    def __init__(self, exciters):
        super().__init__()
        self.exciters = nn.ModuleList(exciters)
        self.alpha = nn.Parameter(torch.zeros(len(exciters)))  # logits
        self.tau = 1.0   # temperature (anneal to ~0.1)
    def weights(self, hard=False):
        if self.training:
            return torch.nn.functional.gumbel_softmax(self.alpha, tau=self.tau, hard=False)
        w = torch.softmax(self.alpha, dim=0)
        return w
    def step_forces(self, t_idx, pos, vel):
        w = self.weights()
        F_total = torch.zeros_like(pos)  # [N,3]
        for k, exc in enumerate(self.exciters):
            node_ids, f_vec = exc(t_idx, pos, vel)   # node_ids: [M], f_vec: (3,)
            F_total.index_add_(0, node_ids, w[k] * f_vec.unsqueeze(0).expand(node_ids.numel(), -1))
        return F_total, w