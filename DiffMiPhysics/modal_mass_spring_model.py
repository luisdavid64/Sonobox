from asyncio import events
import torch
from typing import Optional
from util.audio_helpers import mix_down, rms_normalize
from util.util import event_dict_seconds_to_samples, load_event_from_json
from exciters import *
from diff_mass_spring_model import MassSpringModel


class ModalMassSpringModel(MassSpringModel):
    def __init__(self, nodes, edge_index, springs,
                 drivers=None, listeners=None, config=None,
                 dimX=None, dimY=None, dimZ=None, dist=None,
                 interactionType="FIRST", bounds=[],
                 dt: float = 1/44100, friction: float = 0.25, gain=10):
        super().__init__(nodes, edge_index, springs,
                         drivers=drivers, listeners=listeners, config=config,
                         dimX=dimX, dimY=dimY, dimZ=dimZ, dist=dist,
                         interactionType=interactionType, bounds=bounds,
                         dt=dt, friction=friction, gain=gain)

    def _free_dof_index(self):
        # 1 for free dofs, 0 for fixed
        free_mask_node = (1.0 - self.fixed_mask.view(-1))  # [N]
        free_mask_dof  = free_mask_node.repeat_interleave(3) > 0.5  # [3N]
        free_idx = torch.nonzero(free_mask_dof).view(-1)
        return free_idx

    def _assemble_KZ_dense(self):
        """
        Linearization around rest: F_k ≈ -K x_k - Z (x_k - x_{k-1})
        K,Z ∈ R^{3×3N}, symmetric PSD. No h-scaling here (this is *discrete* damping).
        """
        device, dtype = self._logz.device, self._logz.dtype
        N = self.N
        dofN = 3 * N

        # Edge geometry
        i, j = self.edge_index[0], self.edge_index[1]    # [E]
        pi, pj = self.rest_pos[i], self.rest_pos[j]      # [E,3]
        r  = pj - pi
        # Distance
        L  = (r.pow(2).sum(-1) + 1e-12).sqrt()
        # Directional cosines
        n  = r / L.unsqueeze(-1)                          # [E,3]
        nx, ny, nz = n[:,0], n[:,1], n[:,2]

        # Projection matrix (outer product)
        Pxx = nx*nx; Pxy = nx*ny; Pxz = nx*nz
        Pyx = ny*nx; Pyy = ny*ny; Pyz = ny*nz
        Pzx = nz*nx; Pzy = nz*ny; Pzz = nz*nz

        k_e = self.k   # [E]
        z_e = self.z   # [E]  (discrete “difference” dashpot gain)

        def scatter_axial(val_e):
            Bxx = val_e*Pxx; Bxy = val_e*Pxy; Bxz = val_e*Pxz
            Byx = val_e*Pyx; Byy = val_e*Pyy; Byz = val_e*Pyz
            Bzx = val_e*Pzx; Bzy = val_e*Pzy; Bzz = val_e*Pzz
            dof_i = (i*3).unsqueeze(1) + torch.tensor([0,1,2], device=device)
            dof_j = (j*3).unsqueeze(1) + torch.tensor([0,1,2], device=device)
            ii = dof_i.unsqueeze(2).expand(-1,3,3).reshape(-1)
            ij = dof_i.unsqueeze(2).expand(-1,3,3).reshape(-1)
            ji = dof_j.unsqueeze(2).expand(-1,3,3).reshape(-1)
            jj = dof_j.unsqueeze(2).expand(-1,3,3).reshape(-1)
            ci = dof_i.unsqueeze(1).expand(-1,3,3).reshape(-1)
            cj = dof_j.unsqueeze(1).expand(-1,3,3).reshape(-1)
            ci2= dof_i.unsqueeze(1).expand(-1,3,3).reshape(-1)
            cj2= dof_j.unsqueeze(1).expand(-1,3,3).reshape(-1)
            block = torch.stack([Bxx,Bxy,Bxz, Byx,Byy,Byz, Bzx,Bzy,Bzz], dim=1).reshape(-1)
            return (ii,ci, block), (ij,cj, -block), (ji,ci2, -block), (jj,cj2, block)

        # Assemble global K, Z, size [3N,3N]
        K = torch.zeros((dofN, dofN), device=device, dtype=dtype)
        Z = torch.zeros((dofN, dofN), device=device, dtype=dtype)
        for A, val in ((K, k_e), (Z, z_e)):
            (ii,ci,p), (ij,cj,m1), (ji,ci2,m2), (jj,cj2,p2) = scatter_axial(val)
            A.index_put_((ii,ci), p,  accumulate=True)
            A.index_put_((ij,cj), m1, accumulate=True)
            A.index_put_((ji,ci2), m2, accumulate=True)
            A.index_put_((jj,cj2), p2, accumulate=True)

        # Reduce to free DOFs
        idx = self._free_dof_index()
        Kf = K.index_select(0, idx).index_select(1, idx)
        Zf = Z.index_select(0, idx).index_select(1, idx)
        return Kf, Zf, idx

    def _build_Bu_full(self, drivers: torch.Tensor) -> torch.Tensor:
        device, dtype = self.nodes.device, self.nodes.dtype
        Nd, dofN = int(drivers.numel()), 3*self.N
        Bu = torch.zeros((dofN, 3*Nd), device=device, dtype=dtype)
        for d, node in enumerate(drivers.tolist()):
            base = 3*node
            Bu[base+0, 3*d+0] = 1.0
            Bu[base+1, 3*d+1] = 1.0
            Bu[base+2, 3*d+2] = 1.0
        return Bu

    def _build_C_full(self, listener_ids: torch.Tensor, axis: str='all') -> torch.Tensor:
        device, dtype = self.nodes.device, self.nodes.dtype
        Cn, dofN = int(listener_ids.numel()), 3*self.N
        C = torch.zeros((Cn, dofN), device=device, dtype=dtype)
        for c, node in enumerate(listener_ids.tolist()):
            base = 3*node
            if axis == 'x':
                C[c, base+0] = 1.0
            elif axis == 'y':
                C[c, base+1] = 1.0
            elif axis == 'z':
                C[c, base+2] = 1.0
            else:  # 'all' keeps it linear
                C[c, base+0] = C[c, base+1] = C[c, base+2] = 1.0/3.0
        return C

    def _rasterize_events_to_u(self, T:int, Nd:int, events:dict,
                            device, dtype, hold:int=8,
                            shape:str="cos", driver_axis:str="z"):
        u = torch.zeros(T, 3*Nd, device=device, dtype=dtype)
        if not events: return u
        # envelope
        if hold <= 1:
            env = torch.ones(1, device=device, dtype=dtype)
        else:
            t = torch.arange(hold, device=device, dtype=dtype)
            if shape == "cos":
                env = 0.5 - 0.5*torch.cos(2*torch.pi*(t/(hold-1)))
            elif shape == "exp":
                env = torch.exp(-3.0 * t/(hold-1))
            else:
                env = torch.ones_like(t, dtype=dtype)
        # axis for scalar events
        ax = {"x":0,"y":1,"z":2}.get(driver_axis, 2)
        ax_mask = torch.zeros(3, device=device, dtype=dtype); ax_mask[ax] = 1.0

        for t0, f in events.items():
            t0 = int(t0)
            if t0 >= T: continue
            f = torch.as_tensor(f, device=device, dtype=dtype).view(-1)
            if f.numel()==1: f3 = f[0]*ax_mask
            elif f.numel()>=3: f3 = f[:3]
            else: continue
            fv = f3.view(1,3).expand(Nd,3).reshape(1, 3*Nd)
            t1 = min(T, t0 + env.numel())
            u[t0:t1, :] += env[:t1-t0].view(-1,1) * fv
        return u

    def _init_modal_disc(self):
        if not hasattr(self, "_modal_disc"):
            self._modal_disc = dict()

    def _compute_disc_modes(self, Kf, Zf, idx_free, drivers, listeners, axis, n_modes, diag_gamma=False):
        """
        Diagonalize Kf (M = I here since your inv_mass is scalar), keep k lowest modes.
        Then project Zf into this basis (full Γ for fidelity, or diag only).
        """
        self._init_modal_disc()
        device, dtype = Kf.device, Kf.dtype
        dof_free = Kf.shape[0]
        k = min(n_modes, dof_free)

        # Small/medium -> full eigh; large -> lobpcg
        if dof_free <= 4000:
            w2, U = torch.linalg.eigh(Kf)         # ascending
            w2 = w2[:k]; U = U[:, :k]
        else:
            # LOBPCG for k smallest
            X0 = torch.randn(dof_free, k, device=device, dtype=dtype)
            w2, U = torch.lobpcg(Kf, k=k, B=None, iK=None, niter=100, X=X0, largest=False)
            # sort just in case
            srt = torch.argsort(w2); w2 = w2[srt]; U = U[:, srt]

        # Projections
        Zk = U.T @ (Zf @ U)              # [k,k]
        if diag_gamma:
            Zk = torch.diag(torch.diag(Zk))
        

        # Drivers/listeners (reduce → project)
        Bu_full = self._build_Bu_full(drivers)
        C_full  = self._build_C_full(listeners, axis=axis)
        Bu = Bu_full.index_select(0, idx_free)         # [dof_f, 3*Nd]
        C  = C_full.index_select(1, idx_free)          # [Cl, dof_f]
        Gu = U.T @ Bu                                   # [k, 3*Nd]
        Gy = C @ U                                      # [Cl, k]
    
        det = lambda t: t.detach()
        self._modal_disc.update({
            "U":  det(U),
            "w2": det(w2),
            "Zk": det(Zk),
            "Gu": det(Gu),
            "Gy": det(Gy),
            "idx_free": idx_free,
            "drivers": drivers.detach().clone(),
            "listeners": listeners.detach().clone(),
            "axis": axis,
            "diag_gamma": diag_gamma,
        })

    # ---------- helpers (shared) ----------
    def _prep_modal_diag(self, listener_ids, drivers, axis, n_modes):
        """Build reduced modal model with diagonal Γ; return a small dict of tensors."""
        device, dtype = self.nodes.device, self.nodes.dtype
        listener_ids = self.get_listener_ids() if listener_ids is None else listener_ids
        drivers      = self.get_driver_ids()   if drivers      is None else drivers

        Kf, Zf, idx_free = self._assemble_KZ_dense()
        # cache the large-subspace U0; we will re-diagonalize in that subspace
        self._compute_disc_modes(Kf, Zf, idx_free, drivers, listener_ids, axis, n_modes, diag_gamma=True)

        U0       = self._modal_disc["U"]                # [df, k0]
        idx_free = self._modal_disc["idx_free"]

        # project to subspace and take k lowest
        S_K = U0.T @ (Kf @ U0)                          # [k0,k0]
        S_Z = U0.T @ (Zf @ U0)                          # [k0,k0]
        lam, Vk_full = torch.linalg.eigh(S_K)

        k = min(n_modes, lam.numel())
        w2 = lam[:k]                                     # [k]
        Vk = Vk_full[:, :k].detach()                     # [k0,k]
        U  = (U0 @ Vk)                                   # [df,k]

        # diag Γ entries
        SZV = S_Z @ Vk                                   # [k0,k]
        gamma_vec = (Vk * SZV).sum(dim=0)                # [k]

        # IO projections
        Bu_full = self._build_Bu_full(drivers)           # [3N,3*Nd]
        C_full  = self._build_C_full(listener_ids, axis=axis)  # [C,3N]
        Bu      = Bu_full.index_select(0, idx_free)      # [df,3*Nd]
        C       = C_full.index_select(1, idx_free)       # [C,df]
        Gu      = U.T @ Bu                                # [k,3*Nd]
        Gy      = C  @ U                                  # [C,k]
        Nd      = int(drivers.numel())

        return dict(
            w2=w2, gamma_vec=gamma_vec, Gu=Gu, Gy=Gy,
            Nd=Nd, listener_ids=listener_ids
        )

    def _modal_diag_coeffs(self, w2, gamma_vec):
        """Return per-mode recurrence coefficients a, b for the diag path."""
        alpha = self.inv_mass
        c     = self.inv_mass * self.fric
        a = (2.0 - c) - alpha * (w2 + gamma_vec)         # [k]
        b = -(1.0 - c) + alpha *  gamma_vec              # [k]
        return a, b, alpha

    def _modal_kernel(self, a, b, T, dtype, device):
        """
        Closed-form time kernel g[n] per mode for n=0..T-1 (vectorized).
        g[n] = (r+^n - r-^n) / (r+ - r-)   with double-root fallback.
        """
        n = torch.arange(T, device=device, dtype=dtype).view(T, 1)  # [T,1]
        a_c = a.to(torch.complex64); b_c = b.to(torch.complex64)
        disc = a_c*a_c + 4.0*b_c
        sqrt_disc = torch.sqrt(disc)
        r_plus  = 0.5*(a_c + sqrt_disc)                 # [k]
        r_minus = 0.5*(a_c - sqrt_disc)                 # [k]
        delta   = r_plus - r_minus                      # [k]

        rp = r_plus.unsqueeze(0)  ** n                  # [T,k]
        rm = r_minus.unsqueeze(0) ** n                  # [T,k]

        eps = 1e-6
        near = (torch.abs(delta) < eps)
        delta_safe = torch.where(near, torch.ones_like(delta), delta)
        g_full = (rp - rm) / delta_safe.unsqueeze(0)    # [T,k]

        if near.any():
            r = torch.where(near, r_plus, torch.ones_like(r_plus))
            r_pow_n = r.unsqueeze(0) ** n
            fallback = n * r_pow_n / torch.clamp(r, min=1e-12)
            g_full = torch.where(near.unsqueeze(0), fallback, g_full)

        return g_full.real.to(dtype)                    # [T,k]

    def _build_u0(self, f, Nd, device, dtype):
        """Turn force f into a flat [3*Nd] vector (broadcast [3] → all drivers)."""
        f = torch.as_tensor(f, device=device, dtype=dtype).view(-1)
        if f.numel() == 3:
            return f.unsqueeze(0).expand(Nd, 3).reshape(3*Nd)
        if f.numel() == 3*Nd:
            return f
        raise ValueError(f"Force must be length 3 or 3*Nd (got {f.numel()}).")

    def _finalize_audio(self, y_time_C, listener_ids, layout, pan_method, hp, mix_audio):
        """HPF → mixdown → per-channel peak normalize → [T,K]."""
        if hp:
            y_time_C = self.hp_filter(y_time_C)
        audio = y_time_C
        if mix_audio:
            audio = mix_down(
                self, audio,
                layout=layout, method=pan_method,
                listener_ids=listener_ids,
                normalize=False, soft_clip=False, energy_comp=True,
            )
        audio = audio.transpose(1, 0)  # [K,T]
        peak  = torch.maximum(torch.abs(audio).amax(dim=1), torch.tensor(1e-9, device=audio.device)).detach()
        audio = audio / peak.unsqueeze(-1)
        return audio.T  # [T,K]
    # ---------- end helpers ----------

    def render_modal_impulse_diag(
            self,
            seconds: float,
            fs: int = 16000,
            force_vec=(3.0, 3.0, 3.0),
            listener_ids: Optional[torch.Tensor] = None,
            drivers: Optional[torch.Tensor] = None,
            axis: str = "all",
            n_modes: int = 512,
            hp: bool = True,
            mix_audio: bool = True,
            layout: str = "mono",
            pan_method: str = "by_position",
            start_frame: int = 0,
        ):
        device, dtype = self.nodes.device, self.nodes.dtype
        prep = self._prep_modal_diag(listener_ids, drivers, axis, n_modes)
        a, b, alpha = self._modal_diag_coeffs(prep["w2"], prep["gamma_vec"])

        T = int(round(seconds * fs))
        g_full = self._modal_kernel(a, b, T, dtype, device)              # [T,k]

        u0 = self._build_u0(force_vec, prep["Nd"], device, dtype)         # [3*Nd]
        r0 = alpha * (prep["Gu"] @ u0)                                    # [k]
        q  = g_full * r0.unsqueeze(0)                                     # [T,k]
        y  = q @ prep["Gy"].T                                             # [T,C]
        return self._finalize_audio(y, prep["listener_ids"], layout, pan_method, hp, mix_audio)

    def render_modal_events_diag_overlap_add(
            self,
            seconds: float,
            fs: int = 16000,
            events: dict | None = None,   # {t0_sample: [fx,fy,fz]} or [3*Nd]
            listener_ids: Optional[torch.Tensor] = None,
            drivers: Optional[torch.Tensor] = None,
            axis: str = "all",
            n_modes: int = 512,
            hp: bool = True,
            mix_audio: bool = True,
            layout: str = "mono",
            pan_method: str = "by_position",
        ):
        device, dtype = self.nodes.device, self.nodes.dtype
        events = events or {}

        prep = self._prep_modal_diag(listener_ids, drivers, axis, n_modes)
        a, b, alpha = self._modal_diag_coeffs(prep["w2"], prep["gamma_vec"])

        T = int(round(seconds * fs))
        g_full = self._modal_kernel(a, b, T, dtype, device)              # [T,k]

        C = prep["Gy"].shape[0]
        y_total = torch.zeros((T, C), device=device, dtype=dtype)

        for t0, f in events.items():
            t0 = int(t0)
            if t0 >= T:
                continue
            try:
                u0 = self._build_u0(f, prep["Nd"], device, dtype)        # [3*Nd]
            except ValueError:
                continue
            r0 = alpha * (prep["Gu"] @ u0)                               # [k]
            q  = g_full * r0.unsqueeze(0)                                # [T,k]
            y  = q @ prep["Gy"].T                                        # [T,C]
            dur = T - t0
            if dur > 0:
                y_total[t0:T, :] += y[:dur, :]

        return self._finalize_audio(y_total, prep["listener_ids"], layout, pan_method, hp, mix_audio)

    def render_audio(
            self,
            seconds: float,
            fs: int = 16000,
            observable: str = "pos",      # kept for signature parity
            axis: str = "all",
            listener_ids: Optional[torch.Tensor] = None,
            layout: str = "stereo",
            pan_method: str = "by_position",
            hp: bool = True,
            gain: float | None = None,
            events: dict | None = None,
            exciter=None,
            mix_audio: bool = True,
            start_frame: int = 0,
            n_modes: int = 1024,
            force_vec: tuple[float,float,float] | None = (3.0, 3.0, 3.0),
        ) -> torch.Tensor:
        """
        Unified, fast, no per-sample loop (diag-Γ):
        - If `events` provided -> overlap-add of time-shifted impulses.
        - Else if `force_vec` provided -> single impulse at t=0 with that force.
        - Else -> defaults to single [3,3,3] impulse (change if undesired).
        """
        if events:
            return self.render_modal_events_diag_overlap_add(
                seconds=seconds, fs=fs, events=events,
                listener_ids=listener_ids, drivers=None, axis=axis,
                n_modes=n_modes, hp=hp, mix_audio=mix_audio,
                layout=layout, pan_method=pan_method,
            )
        # single impulse path
        fvec = (3.0,3.0,3.0) if force_vec is None else force_vec
        return self.render_modal_impulse_diag(
            seconds=seconds, fs=fs, force_vec=fvec,
            listener_ids=listener_ids, drivers=None, axis=axis,
            n_modes=n_modes, hp=hp, mix_audio=mix_audio,
            layout=layout, pan_method=pan_method,
            start_frame=start_frame,
        )



if __name__ == "__main__":
    fs = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModalMassSpringModel.from_json(
        "../model_configs/sonobox_data/baselines/biosonix_3D.json",
        device=device, dt=1/fs
    )
    model.train()  # enable grads

    # # Render 1s of audio and backprop a simple power loss
    seconds = 6.0
    events = load_event_from_json("events/two_hits.json")
    events = event_dict_seconds_to_samples(events, fs)

    audio = model.render_modal_events_diag_overlap_add(
        seconds=seconds,
        fs=fs,
        events=events,
        listener_ids=model.get_listener_ids(),
        drivers=model.get_driver_ids(),
        axis='all',
        n_modes=1024,
        hp=True,
        mix_audio=True,
        layout='mono',
        pan_method='by_position',
    )

    audio = audio.squeeze()
    
    import soundfile as sf, sounddevice as sd
    sf.write("mass_spring.wav", audio.detach().cpu().numpy(), fs)
    sd.play(audio.detach().cpu().numpy(), fs); sd.wait()

    loss = torch.mean(audio**2)
    print("Audio loss:", loss.item())
    loss.backward()
