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
        

    # @torch.compile()
    def render_modal_audio(self,
                        seconds: float,
                        fs: int = 16000,
                        listener_ids: Optional[torch.Tensor] = None,
                        drivers: Optional[torch.Tensor] = None,
                        axis: str = "all",
                        n_modes: int = 512,
                        gamma: str = "full",    # 'full' or 'diag'
                        hp: bool = True,
                        events: dict = {},
                        mix_audio: bool = True,
                        layout: str = "mono",
                        pan_method: str = "by_position",
                        start_frame: int = 0):
        """
        Modal rendering with toggleable damping:
        - gamma='full': uses full Γ = U^T Z U (k×k) → O(k^2) per step
        - gamma='diag': uses only diag(Γ)        → O(k) per step
        """
        device, dtype = self.nodes.device, self.nodes.dtype
        listener_ids = self.get_listener_ids() if listener_ids is None else listener_ids
        drivers      = self.get_driver_ids()   if drivers      is None else drivers
        diag_gamma   = (gamma == "diag")

        # 1) Assemble reduced K, Z once (you can later switch this to linear-combo bases)
        Kf, Zf, idx_free = self._assemble_KZ_dense()

        # if need_modes:
        #     # This fills a cache with U (subspace), etc. You can leave it as-is.
        self._compute_disc_modes(Kf, Zf, idx_free, drivers, listener_ids, axis, n_modes, diag_gamma=diag_gamma)

        # --- Build a *training-safe* subspace projection ---
        # If you want grads through U, avoid .detach(); if you only train scalars (k,z,fric), detaching is fine.
        U0       = self._modal_disc["U"]         # [df, k0] cached subspace (k0 >= k)
        idx_free = self._modal_disc["idx_free"]

        # Project current K, Z into U0 (small k0×k0)
        S_K = U0.T @ (Kf @ U0)                   # [k0, k0]
        S_Z = U0.T @ (Zf @ U0)                   # [k0, k0]

        # Small eigendecomp → the first k modes
        lam, Vk_full = torch.linalg.eigh(S_K)    # ascending
        k = min(n_modes, lam.numel())
        w2 = lam[:k]                             # [k]
        # Vk = Vk_full[:, :k].detach()                      # [k0, k]
        Vk = Vk_full[:, :k].detach()                      # [k0, k]
        U  = (U0 @ Vk)                             # [df, k] final modal basis

        # Damping representation
        if diag_gamma:
            # diag(Γ) = diag( Vk^T S_Z Vk )  without forming full k×k Γ
            SZV = S_Z @ Vk                      # [k0, k]
            gamma_vec = (Vk * SZV).sum(dim=0)   # [k]
            Zk = None
        else:
            # full Γ in k×k space
            Zk = Vk.T @ S_Z @ Vk                # [k, k]
            gamma_vec = None

        # Drivers/listeners projection
        Bu_full = self._build_Bu_full(drivers)           # [3N, 3*Nd]
        C_full  = self._build_C_full(listener_ids, axis=axis)  # [C, 3N]
        Bu      = Bu_full.index_select(0, idx_free)      # [df, 3*Nd]
        C       = C_full.index_select(1, idx_free)       # [C, df]

        Gu = U.T @ Bu                                    # [k, 3*Nd]
        Gy = C  @ U                                      # [C, k]

        # 3) Two-step coefficients
        alpha = self.inv_mass                            # scalar
        c     = self.inv_mass * self.fric                # scalar

        if diag_gamma:
            # elementwise recurrence: q_{n+1} = a*q_n + b*q_{n-1} + (Uu @ u_n)
            a = (2.0 - c) - alpha * (w2 + gamma_vec)     # [k]
            b = -(1.0 - c) + alpha *  gamma_vec          # [k]
            Uu = alpha * Gu                              # [k, 3*Nd]
        else:
            # coupled recurrence with dense A,B
            I = torch.eye(k, device=device, dtype=dtype)
            Λ = torch.diag(w2)                           # [k, k]
            A = (2.0 - c) * I - alpha * (Λ + Zk)         # [k, k]
            B = -(1.0 - c) * I + alpha * Zk              # [k, k]
            Uu = alpha * Gu                              # [k, 3*Nd]

        # 4) Inputs (you can switch to on-the-fly later to save memory)
        T  = int(round(seconds * fs))
        Nd = drivers.numel()
        u  = self._rasterize_events_to_u(T=T, Nd=Nd, events=events, device=device, dtype=dtype,
                                        hold=1, shape="cos", driver_axis="z")  # [T, 3*Nd]

        # 5) State & render
        q_prev = torch.zeros(k, device=device, dtype=dtype)
        q      = torch.zeros(k, device=device, dtype=dtype)
        y      = torch.empty((T, Gy.shape[0]), device=device, dtype=dtype)  # [T, C]

        if diag_gamma:
            # O(k) per step
            for t in range(T):
                r_t   = Uu @ u[t]                    # [k]
                q_next = a * q + b * q_prev + r_t    # [k]
                y[t]   = Gy @ q_next                 # [C]
                q_prev, q = q, q_next
        else:
            # O(k^2) per step
            for t in range(T):
                r_t   = Uu @ u[t]                    # [k]
                q_next = A @ q + B @ q_prev + r_t    # [k]
                y[t]   = Gy @ q_next                 # [C]
                q_prev, q = q, q_next

        # 6) Optional HPF
        if hp:
            y = self.hp_filter(y)                    # [T, C]

        # 7) Mixdown & normalize
        audio = y
        if mix_audio:
            audio = mix_down(
                self, audio,
                layout=layout, method=pan_method,
                listener_ids=listener_ids,
                normalize=False, soft_clip=False, energy_comp=True,
            )
        audio = audio.transpose(1, 0)  # [K, T]
        # audio = rms_normalize(audio)  # normalize to -12 dBFS
        peak  = torch.maximum(torch.abs(audio).amax(dim=1), torch.tensor(1e-9, device=audio.device)).detach()
        audio = audio / peak.unsqueeze(-1)
        return audio.T  # [T, K]

    # def render_audio(self, seconds: float, fs: int = 16000, observable: str = "pos", axis: str = "all", listener_ids: torch.Tensor | None = None, layout: str = "stereo", pan_method: str = "by_position", hp: bool = True, gain: float | None = None, events: dict = ..., exciter=None, mix_audio: bool = True, start_frame=0) -> torch.Tensor:
    #     return self.render_modal_audio(seconds, fs, n_modes=1500, gamma="full", listener_ids=listener_ids, layout=layout, axis=axis, events=events, pan_method=pan_method, hp=hp, mix_audio=mix_audio, start_frame=start_frame)

    def render_modal_impulse_diag(
            self,
            seconds: float,
            fs: int = 16000,
            force_vec=(3.0, 3.0, 3.0),   # impulse @ t=0 applied to every driver
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
        """
        Vectorized impulse response for gamma='diag' (no Python time loop).

        Implements the closed-form solution of:
            q_{n+1} = a ⊙ q_n + b ⊙ q_{n-1}       (no input for n>=1)
        with impulse only at n=0:
            q_0 = 0
            q_1 = r0 = (alpha * Gu @ u0)

        Closed form per mode m:
            q_n = r0_m * (r+_m^n - r-_m^n) / (r+_m - r-_m)
        (and if r+≈r- we fallback to q_n = r0_m * n * r^ {n-1}).

        Returns: [T, K] audio tensor (time-major), like render_modal_audio.
        """
        device, dtype = self.nodes.device, self.nodes.dtype
        listener_ids = self.get_listener_ids() if listener_ids is None else listener_ids
        drivers      = self.get_driver_ids()   if drivers      is None else drivers

        # 1) Assemble K, Z (free DOFs) and compute modal subspace with diag Γ
        Kf, Zf, idx_free = self._assemble_KZ_dense()

        # Reuse the cached subspace machinery (diag_gamma=True)
        self._compute_disc_modes(
            Kf, Zf, idx_free,
            drivers, listener_ids, axis,
            n_modes, diag_gamma=True
        )

        # Cached/constructed bits
        U0       = self._modal_disc["U"]                # [df, k0]
        idx_free = self._modal_disc["idx_free"]

        # Project current K, Z to subspace, then pick k smallest modes
        S_K = U0.T @ (Kf @ U0)                          # [k0, k0]
        S_Z = U0.T @ (Zf @ U0)                          # [k0, k0]
        lam, Vk_full = torch.linalg.eigh(S_K)
        k = min(n_modes, lam.numel())
        w2 = lam[:k]                                     # [k]
        Vk = Vk_full[:, :k].detach()                     # [k0, k]
        U  = (U0 @ Vk)                                   # [df, k]

        # Diagonal Γ entries in modal coords
        SZV = S_Z @ Vk                                   # [k0, k]
        gamma_vec = (Vk * SZV).sum(dim=0)                # [k]

        # Driver/listener projection
        Bu_full = self._build_Bu_full(drivers)           # [3N, 3*Nd]
        C_full  = self._build_C_full(listener_ids, axis=axis)  # [C, 3N]
        Bu      = Bu_full.index_select(0, idx_free)      # [df, 3*Nd]
        C       = C_full.index_select(1, idx_free)       # [C, df]
        Gu      = U.T @ Bu                                # [k, 3*Nd]
        Gy      = C  @ U                                  # [C, k]

        # 2) Recurrence coefficients (diag path)
        alpha = self.inv_mass                             # scalar
        c     = self.inv_mass * self.fric                 # scalar
        a = (2.0 - c) - alpha * (w2 + gamma_vec)          # [k]
        b = -(1.0 - c) + alpha *  gamma_vec               # [k]

        # 3) Build the single-sample impulse u0 (applied to every driver)
        Nd = int(drivers.numel())
        f3 = torch.as_tensor(force_vec, device=device, dtype=dtype).view(3)
        u0 = f3.unsqueeze(0).expand(Nd, 3).reshape(3*Nd)   # [3*Nd]

        # Modal "kick" at n=1: r0 = (alpha * Gu @ u0)
        r0 = (alpha * (Gu @ u0))                           # [k]

        # 4) Closed-form q_n for n = 0..T-1 (vectorized)
        T = int(round(seconds * fs))
        n = torch.arange(T, device=device, dtype=dtype).view(T, 1)     # [T,1]

        # Roots of r^2 - a r - b = 0 (use complex to be robust)
        a_c = a.to(torch.complex64)
        b_c = b.to(torch.complex64)
        disc = a_c*a_c + 4.0*b_c
        sqrt_disc = torch.sqrt(disc)
        r_plus  = 0.5*(a_c + sqrt_disc)      # [k]
        r_minus = 0.5*(a_c - sqrt_disc)      # [k]
        delta   = r_plus - r_minus            # [k]

        # Powers r^n, broadcast to [T,k]
        rp = r_plus.unsqueeze(0)  ** n
        rm = r_minus.unsqueeze(0) ** n

        # Handle near-double-root modes stably
        eps = 1e-6
        near = (torch.abs(delta) < eps)
        delta_safe = torch.where(near, torch.ones_like(delta), delta)

        # General case
        q_full = (r0.to(torch.complex64).unsqueeze(0) * (rp - rm) / delta_safe)  # [T,k]

        # Near-double-root fallback: q_n = r0 * n * r^{n-1}
        if near.any():
            r = torch.where(near, r_plus, torch.ones_like(r_plus))                # [k]
            # n * r^{n-1} = n * (r^n) / r
            r_pow_n = r.unsqueeze(0) ** n                                         # [T,k]
            fallback = (r0.to(torch.complex64).unsqueeze(0) *
                        (n * r_pow_n / torch.clamp(r, min=1e-12)))
            q_full = torch.where(near.unsqueeze(0), fallback, q_full)

        # Real part (the recurrence and parameters are real → result must be real)
        q_full = q_full.real.to(dtype)                                            # [T,k]

        # 5) Outputs in listener space: y = Q @ Gy^T
        y = q_full @ Gy.T                                                          # [T, C]

        # 6) Optional HPF and mixdown
        if hp:
            y = self.hp_filter(y)                                                 # [T, C]

        audio = y
        if mix_audio:
            audio = mix_down(
                self, audio,
                layout=layout, method=pan_method,
                listener_ids=listener_ids,
                normalize=False, soft_clip=False, energy_comp=True,
            )

        # Normalize by per-channel peak (like your render)
        audio = audio.transpose(1, 0)  # [K, T]
        peak  = torch.maximum(torch.abs(audio).amax(dim=1), torch.tensor(1e-9, device=audio.device)).detach()
        audio = audio / peak.unsqueeze(-1)
        return audio.T  # [T, K]

    def render_audio(self, seconds: float, fs: int = 16000, observable: str = "pos", axis: str = "all", listener_ids: torch.Tensor | None = None, layout: str = "stereo", pan_method: str = "by_position", hp: bool = True, gain: float | None = None, events: dict = ..., exciter=None, mix_audio: bool = True, start_frame=0) -> torch.Tensor:
        return self.render_modal_impulse_diag(seconds, fs, force_vec=(3,3,3), n_modes=1500, listener_ids=listener_ids, layout=layout, axis=axis, hp=hp, mix_audio=mix_audio, start_frame=start_frame)

    def render_modal_events_diag_overlap_add(
            self,
            seconds: float,
            fs: int = 16000,
            events: dict | None = None,   # {t0_sample: [fx,fy,fz]} or [3*Nd] per event
            listener_ids: Optional[torch.Tensor] = None,
            drivers: Optional[torch.Tensor] = None,
            axis: str = "all",
            n_modes: int = 512,
            hp: bool = True,
            mix_audio: bool = True,
            layout: str = "mono",
            pan_method: str = "by_position",
        ):
        """
        LTI overlap-add for multiple impulses (diag Γ), no per-sample loop.
        Each events[t0] is applied at sample index t0 and broadcast to all drivers
        if it’s a 3-vector; or you can pass a full per-driver vector of length 3*Nd.
        Returns [T, K] audio (time-major).
        """
        device, dtype = self.nodes.device, self.nodes.dtype
        listener_ids = self.get_listener_ids() if listener_ids is None else listener_ids
        drivers      = self.get_driver_ids()   if drivers      is None else drivers
        events = events or {}

        # --- 1) Assemble reduced model and modal projections (diag Γ) ---
        Kf, Zf, idx_free = self._assemble_KZ_dense()
        self._compute_disc_modes(Kf, Zf, idx_free, drivers, listener_ids, axis, n_modes, diag_gamma=True)

        U0       = self._modal_disc["U"]                 # [df, k0]
        idx_free = self._modal_disc["idx_free"]

        S_K = U0.T @ (Kf @ U0)                           # [k0,k0]
        S_Z = U0.T @ (Zf @ U0)                           # [k0,k0]
        lam, Vk_full = torch.linalg.eigh(S_K)
        k = min(n_modes, lam.numel())
        w2 = lam[:k]                                     # [k]
        Vk = Vk_full[:, :k].detach()                     # [k0,k]
        U  = (U0 @ Vk)                                   # [df,k]

        # diag Γ
        SZV = S_Z @ Vk                                   # [k0,k]
        gamma_vec = (Vk * SZV).sum(dim=0)                # [k]

        # I/O projections
        Bu_full = self._build_Bu_full(drivers)           # [3N, 3*Nd]
        C_full  = self._build_C_full(listener_ids, axis=axis)  # [C,3N]
        Bu      = Bu_full.index_select(0, idx_free)      # [df,3*Nd]
        C       = C_full.index_select(1, idx_free)       # [C,df]
        Gu      = U.T @ Bu                                # [k,3*Nd]
        Gy      = C  @ U                                  # [C,k]

        # --- 2) Precompute per-mode impulse kernel g[n] (closed form) ---
        alpha = self.inv_mass
        c     = self.inv_mass * self.fric
        a = (2.0 - c) - alpha * (w2 + gamma_vec)         # [k]
        b = -(1.0 - c) + alpha *  gamma_vec              # [k]

        T = int(round(seconds * fs))
        n = torch.arange(T, device=device, dtype=dtype).view(T, 1)  # [T,1]

        a_c = a.to(torch.complex64); b_c = b.to(torch.complex64)
        disc = a_c*a_c + 4.0*b_c
        sqrt_disc = torch.sqrt(disc)
        r_plus  = 0.5*(a_c + sqrt_disc)                  # [k]
        r_minus = 0.5*(a_c - sqrt_disc)                  # [k]
        delta   = r_plus - r_minus                       # [k]

        rp = r_plus.unsqueeze(0)  ** n                   # [T,k]
        rm = r_minus.unsqueeze(0) ** n                   # [T,k]

        eps = 1e-6
        near = (torch.abs(delta) < eps)
        delta_safe = torch.where(near, torch.ones_like(delta), delta)

        # General closed-form kernel g[n] = (r+^n - r-^n) / (r+ - r-)
        g_full = (rp - rm) / delta_safe.unsqueeze(0)     # [T,k]

        # Double-root fallback: g[n] = n * r^(n-1)
        if near.any():
            r = torch.where(near, r_plus, torch.ones_like(r_plus))
            r_pow_n = r.unsqueeze(0) ** n                # [T,k]
            fallback = n * r_pow_n / torch.clamp(r, min=1e-12)
            g_full = torch.where(near.unsqueeze(0), fallback, g_full)

        g_full = g_full.real.to(dtype)                   # [T,k]

        # --- 3) Overlap-add per event (time-shifted impulse responses) ---
        C = Gy.shape[0]
        y_total = torch.zeros((T, C), device=device, dtype=dtype)

        Nd = int(drivers.numel())
        for t0, f in events.items():
            t0 = int(t0)
            if t0 >= T:
                continue

            f = torch.as_tensor(f, device=device, dtype=dtype).view(-1)
            if f.numel() == 3:
                # broadcast same [fx,fy,fz] to all drivers
                u0 = f.unsqueeze(0).expand(Nd, 3).reshape(3*Nd)
            elif f.numel() == 3*Nd:
                u0 = f
            else:
                # ignore malformed force
                continue

            # modal kick at n=1 for this event
            r0 = alpha * (Gu @ u0)                       # [k]

            # q_event[n] = r0 ⊙ g_full[n]
            q_event = g_full * r0.unsqueeze(0)           # [T,k] (but we will trim)
            y_event = q_event @ Gy.T                     # [T,C]

            # time shift (overlap-add)
            dur = T - t0
            if dur > 0:
                y_total[t0:T, :] += y_event[:dur, :]

        # --- 4) HPF, mix, normalize ---
        if hp:
            y_total = self.hp_filter(y_total)

        audio = y_total
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



if __name__ == "__main__":
    fs = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModalMassSpringModel.from_json(
        "../model_configs/sonobox_data/baselines/biosonix_3D.json",
        device=device, dt=1/fs
    )
    model.train()  # enable grads

    # # Render 1s of audio and backprop a simple power loss
    seconds = 1.0
    events = load_event_from_json("events/two_hits.json")
    events = event_dict_seconds_to_samples(events, fs)

    # audio = model.render_modal_audio(
    #     seconds=1.0,
    #     fs=fs,
    #     listener_ids=model.get_listener_ids(),
    #     drivers=model.get_driver_ids(),
    #     axis='all',          # 'x'|'y'|'z'|'all' (linear; 'all' ≈ your 'all')
    #     n_modes=1024,         # keep the most audible modes
    #     hp=True,
    #     events=events,       # same events dict you already use
    #     mix_audio=True,
    #     gamma="diag",
    #     layout='mono',
    #     pan_method='by_position',
    # )

    # audio = model.render_modal_impulse_diag(
    #     seconds=1.0,
    #     fs=fs,
    #     force_vec=(3.0, 3.0, 3.0),
    #     listener_ids=model.get_listener_ids(),
    #     drivers=model.get_driver_ids(),
    #     axis='all',
    #     n_modes=1024,
    #     hp=True,
    #     mix_audio=True,
    #     layout='mono',
    #     pan_method='by_position',
    # )

    audio = model.render_modal_events_diag_overlap_add(
        seconds=1.0,
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
