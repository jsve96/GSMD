import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision
import numpy as np
import skimage
import math
import bisect
from tqdm import tqdm
from skimage.segmentation import slic

class Attack(object):
    '''
    Root class for all adversarial attack classes.
    '''

    def __init__(self, model, targeted=False, img_range=(0, 1)):
        self.model = model
        self.device = next(model.parameters()).device
        self.targeted = targeted
        self.img_range = img_range

    def __repr__(self):
        return str(self.__dict__)

    def to(self, device):
        self.model.to(device)
        self.device = device

##################################### GSMD #####################################

class GSMD(Attack):
    """
    Group-wise sparse attack using mirror descent (GSMD).

    This attack optimizes an adversarial perturbation under
    group-structures using mirror descent.
    """

    def __init__(
        self,
        model,
        *args,
        img_range=(-1, 1),
        targeted: bool = True,
        sequential: bool = False,
        D: float = 1.0,
        l: float = 1.0,
        u: float = 1.0,
        learning_rate = 1.0,
        rho = 0.1,
        patches = 30,
        debug = False,
        **kwargs
    ):
        """
        Initialize the GSMD attack.

        Args:
            model (torch.nn.Module): Target model.
            ver (bool): Verbosity flag.
            img_range (tuple): Allowed pixel range.
            targeted (bool): Whether the attack is targeted.
            sequential (bool): Run attack sequentially per sample.
            D (float): Budget for projection constraint.
            l (float): Lower bound for perturbation.
            u (float): Upper bound for perturbation.
        """
        super().__init__(model, img_range=img_range)
        self.sequential = sequential
        self.D = D
        self.l = l
        self.u = u
        self.targeted = targeted
        self.eta=0.0
        self.learning_rate = learning_rate
        self.rho = rho
        self.patches = patches
        self.debug = debug

    def __call__(self, x, y, *args, **kwargs):
        """
        Run the attack on a batch of images.

        Depending on the `sequential` flag, the attack is applied
        either per-sample or to the full batch at once.

        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W].
            y (torch.Tensor): Labels of shape [B].

        Returns:
            torch.Tensor: Adversarial examples of same shape as x.
        """
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.perform_att(
                    x_.unsqueeze(0),
                    y_.unsqueeze(0),
                    D = self.D, 
                    l = self.l,
                    u = self.u
                    ).detach()
            return result
        else:
            print('Not sequential')
            return self.perform_att(x, y, D=self.D, l=self.l, u=self.u,rho=self.rho,patches=self.patches,debug=self.debug)

    def perform_att(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        D: float = 5.0,
        l: float = None,
        u: float = None,
        steps: int = 200,
        patches = 15,
        rho = 0.1,
        debug = False
    ) -> torch.Tensor:
        """
        Perform the attack using mirror descent.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
            y (torch.Tensor): Ground-truth labels [B].
            D (float): Constraint budget.
            l (float): Lower bound on perturbation.
            u (float): Upper bound on perturbation.
            steps (int): Number of optimization steps.

        Returns:
            torch.Tensor: Adversarial images.
        """
       
        u = 1-x
        l = -1-x
        self.eta=0.0

        self.B, self.C, self.H, self.W = x.shape
        print(D)
        self.epsilon = torch.full((self.B,), D, device=self.device) # epsilon budget per image

        D_MAX = D * 2.0
        D_MIN = 0.001 
        print(rho)

        self.max_groups = int(10/22.5*patches**2) 
        self.group_budget = self.max_groups * torch.ones(self.B, device=self.device)      # exact group sparsity
        self.alpha = 0.5# 1.0#0.5

        segmentation = self.get_grid_segmentation(n=patches)

        delta = torch.zeros_like(x, requires_grad=True)

        x_adv = x + delta 

        delta_best = torch.zeros_like(x_adv)

        clean_labels = torch.argmax(self.model(x),dim=1)

        LOSSES_ = []
        T_warm = int(self.alpha * steps)

        for t in range(steps):
            logits = self.model(x_adv)
            current_max_labels = torch.argmax(logits,dim=1)

            target_logit = logits[torch.arange(self.B), y] #new
            true_logit = logits[torch.arange(self.B), current_max_labels]

            if self.targeted:
                losses = F.cross_entropy(logits, y, reduction="none")
                margin = target_logit - true_logit #
                increase = margin < 0.0
                decrease = margin >= 0.0
            else:
                losses = -F.cross_entropy(logits, y, reduction="none")
                margin = true_logit - target_logit
                increase = margin <= 0.0
                decrease = margin > 0.0

            grads = torch.autograd.grad(
                losses.sum(), delta, retain_graph=False, create_graph=False
            )[0]
            with torch.no_grad():

                probs = F.softmax(logits, dim=1)
                conf = probs[torch.arange(self.B), y]
                s_b = 1.0 + (self.max_groups - 1.0) * (1 - conf)
                self.group_budget = s_b.round().long()

                ##### NEW ######
                self.epsilon[increase] *= (1 + rho)
                self.epsilon[decrease] *= (1 - rho)
                self.epsilon = torch.clamp(self.epsilon, min=D_MIN)

                ###### NEW #####


                new_delta = self._md(g=grads,x=delta,lower=l,upper=u,beta=1.0,grid=segmentation)
                
                if t == T_warm:
                    self.active_groups = None 

                if t >= T_warm:
                    new_delta = self._persistent_group_threshold(new_delta,segmentation)

                delta.copy_(new_delta)

            LOSSES_.append(losses.detach().cpu())

            
            x_adv = x + delta
            if self.targeted:
                mask_prediction = torch.argmax(self.model(x_adv), dim=1) == y
                delta_best[mask_prediction] = new_delta[mask_prediction]
            else:
                mask_prediction = torch.argmax(self.model(x_adv), dim=1) != y
                l0s = torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2))
                if t == 0:
                    l02_best = l0s
                else:
                    old_delta = delta_best
                    delta_best[mask_prediction] = new_delta[mask_prediction]
                    mask_l0 = l02_best > l0s
                    l02_best[mask_l0] = l0s[mask_l0]
                    mask_prediction = mask_prediction * mask_l0
                    delta_best[mask_prediction] = old_delta[mask_prediction]


                
            #print(mask_prediction)
            #delta.requires_grad = True
        if debug:
            (x + delta_best).detach(),LOSSES_

        return (x + delta_best).detach()


    def _md(self, g: torch.Tensor, x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, beta: float,grid:torch.Tensor) -> torch.Tensor:
        # 1. Map to Dual Space
        dual_x = torch.log(torch.abs(x) / beta + 1.0) * torch.sign(x)
        
        max_g2 = torch.amax(g**2, dim=(1, 2, 3)) 
        self.eta += max_g2[:, None, None, None]
        
        eta_t = torch.sqrt(self.eta) / self.learning_rate
        descent = g / eta_t
        
        # 3. Update in Dual Space
        z = dual_x - descent
        z_sgn = torch.sign(z)
        z_val = torch.abs(z)
        
        # 4. Projection (looping over batch dimension)
        v_list = []
    
        for d in range(dual_x.shape[0]):
            v_list.append(self._project(z_sgn[d], z_val[d], beta, self.epsilon[d], lower[d], upper[d],grid))
        
        return torch.stack(v_list, dim=0)



    def _project(self, y_sgn_full: torch.Tensor, dual_y_val_full: torch.Tensor, beta: float, D: float, l_full: torch.Tensor, u_full: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        c_full = torch.where(y_sgn_full <= 0, torch.abs(l_full), u_full)
        dual_c_full = torch.log(c_full / beta + 1.0)
        
        # target represents min(y, c)
        target = torch.minimum(dual_y_val_full, dual_c_full)
        C, H, W = target.shape
        num_groups = grid.max().item() + 1

        # flat_target shape: (C, H*W)
        flat_target = target.view(C, -1).reshape(-1)
        #print(flat_target.shape)
        # flat_grid shape: (C, H*W) -> every channel follows same grid
        #flat_grid = grid.view(-1).expand(C, -1)
        full_grid = grid.expand(C, H, W).reshape(-1)

        group_winner_val = torch.zeros(num_groups, device=target.device).scatter_reduce(0, full_grid, flat_target, reduce='amax')

        M_ = (grid-grid.min().item()).view(-1) # H*W

        possible_winners = (target == group_winner_val[grid]) 

        flat_possible = possible_winners.view(-1)
        final_indices = torch.zeros(num_groups, dtype=torch.long, device=target.device)
        final_indices = final_indices.scatter(0, full_grid[flat_possible], torch.arange(C*H*W, device=target.device)[flat_possible])

        dual_y_val = dual_y_val_full.view(-1)[final_indices]
        dual_c = dual_c_full.view(-1)[final_indices]
        c = c_full.view(-1)[final_indices]
        


        # Check if budget D is already satisfied
        phi_0 = beta * torch.exp(torch.clamp(torch.minimum(dual_y_val, dual_c), min=0.0)) - beta
        if torch.sum(phi_0) <= D:
            v = phi_0
        else:
            # Binary Search for Lagrange Multiplier
            z = torch.cat(((dual_y_val).reshape(-1), (dual_y_val - dual_c).reshape(-1)))
            z = torch.sort(z[z >= 0])[0]
            
            idx_l, idx_u = 0, len(z) - 1
            while idx_u - idx_l > 1:
                idx = (idx_u + idx_l) // 2
                lam = z[idx]
                phi = torch.sum(beta * torch.exp(torch.clamp(torch.minimum(dual_y_val - lam, dual_c), min=0.0)) - beta)
                if phi > D:
                    idx_l = idx
                else:
                    idx_u = idx
                    
            lam_lower, lam_upper = z[idx_u], z[idx_l]
            
            
            lam = (lam_lower + lam_upper) / 2.0
            idx_clip = (dual_y_val - lam) >= dual_c
            idx_active = ((dual_y_val - lam) < dual_c) & ((dual_y_val - lam) > 0)
            
            v = torch.where(idx_clip, c, torch.zeros_like(c))
            num_active = torch.sum(idx_active)
            if num_active != 0:
                sum_active = D - torch.sum(c[idx_clip])
                normaliser = (sum_active + beta * num_active) / torch.sum(beta * torch.exp(dual_y_val)[idx_active])
                v = torch.where(idx_active, beta * torch.exp(dual_y_val) * normaliser - beta, v)

        _y_val_full = beta * torch.exp(torch.minimum(dual_y_val_full, dual_c_full)) - beta

        v = v[M_].view(self.H,self.W).expand(self.C,-1,-1)

        v_full = torch.minimum(_y_val_full, v)
        return v_full * y_sgn_full


    def get_grid_segmentation(self, n):
        """
        Creates an (H, W) tensor where pixels are grouped into an n x n grid.
        
        Args:
            H, W: Height and Width of the image.
            n: Number of blocks along each dimension (total groups = n*n).
        """
        block_h = self.H // n
        block_w = self.W // n
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=self.device), 
            torch.arange(self.W, device=self.device), 
            indexing='ij'
        )
        
        row_idx = torch.div(y_coords, block_h, rounding_mode='floor').clamp(max=n)
        col_idx = torch.div(x_coords, block_w, rounding_mode='floor').clamp(max=n)
        
        grid_indices = row_idx * n + col_idx
        
        return grid_indices.long()


    def _persistent_group_threshold(self, delta, grid):
        B, C, H, W = delta.shape
        num_groups = grid.max().item() + 1
        device = delta.device

        if self.active_groups is None:
            self.active_groups = []

            full_grid = grid.unsqueeze(0).expand(C, H, W).reshape(-1)

            for b in range(B):
                abs_delta = delta[b].abs().reshape(-1)

                scores = torch.zeros(num_groups, device=device)
                scores.scatter_reduce_(0, full_grid, abs_delta, reduce="amax")

                s_b = int(self.group_budget[b].item())
                _, active = torch.topk(scores, s_b)

                self.active_groups.append(active)

        out = torch.zeros_like(delta)

        for b in range(B):
            group_mask = torch.zeros(num_groups, device=device)
            group_mask[self.active_groups[b]] = 1.0

            pixel_mask = group_mask[grid].unsqueeze(0).expand(C, -1, -1)
            out[b] = delta[b] * pixel_mask

        return out
    
#################################### GSMD-exactprojection ##################################


class GSMD_exact_new(Attack):
    def __init__(
        self,
        model,
        *args,
        img_range=(-1, 1),
        targeted: bool = True,
        sequential: bool = False,
        D: float = 1.0,
        l: float = 1.0,
        u: float = 1.0,
        learning_rate=1.0,
        rho=0.1,
        patches=30,
        steps = 200,
        debug = False,
        **kwargs
    ):
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.sequential = sequential
        self.D = D
        self.l = l
        self.u = u
        self.targeted = targeted
        self.eta = 0.0
        self.learning_rate = learning_rate
        self.rho = rho
        self.patches = patches
        self.debug = debug
        self.steps = steps
        self.active_groups = None


    def __call__(self, x, y, *args, **kwargs):
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.perform_att(
                    x_.unsqueeze(0),
                    y_.unsqueeze(0),
                    D=self.D,
                    l=self.l,
                    u=self.u,
                    rho=self.rho,
                    patches=self.patches,
                ).detach()
            return result

        return self.perform_att(
            x,
            y,
            D=self.D,
            l=self.l,
            u=self.u,
            rho=self.rho,
            patches=self.patches, debug = self.debug,steps=self.steps
        )

    def perform_att(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        D: float = 5.0,
        l: float = None,
        u: float = None,
        steps: int = 200,
        patches=15,
        rho=0.1,
        debug = False
    ) -> torch.Tensor:

        # Box constraints for delta so that x + delta stays in [-1, 1]
        upper = 1 - x
        lower = -1 - x
        print(steps)

        print(D)
        self.eta = 0.0
        self.B, self.C, self.H, self.W = x.shape

        self.epsilon = torch.full(
            (self.B,),
            D,
            device=x.device,
            dtype=x.dtype,
        )

        D_MIN = 0.001

        self.max_groups = int(10 / 22.5 * patches**2)
        self.group_budget = self.max_groups * torch.ones(
            self.B,
            device=x.device,
            dtype=torch.long,
        )

        self.alpha = 0.5# 1.0 #0.5
        T_warm = int(self.alpha * int(steps))

        # Important: keep both structures.
        # grid is [H, W], used for thresholding.
        # groups is a list of flat [C,H,W] indices, used for projection.
        grid = self.get_grid_segmentation(n=patches)
        print(grid)
        LOSSES_ = []

        delta = torch.zeros_like(x, requires_grad=True)
        x_adv = x + delta
        delta_best = torch.zeros_like(x)

        with torch.no_grad():
            clean_labels = torch.argmax(self.model(x), dim=1)

        self.active_groups = None

        for t in tqdm(range(steps)):
            logits = self.model(x_adv)
            current_max_labels = torch.argmax(logits,dim=1)

            target_logit = logits[torch.arange(self.B), y] #new
            true_logit = logits[torch.arange(self.B), current_max_labels] # vorher logits[torch.arange(self.B),clean_labels] #new


            if self.targeted:
                losses = F.cross_entropy(logits, y, reduction="none")
                margin = target_logit - true_logit #
                increase = margin < 0.0
                decrease = margin >= 0.0
            else:
                losses = -F.cross_entropy(logits, y, reduction="none")
                margin = true_logit - target_logit
                increase = margin <= 0.0
                decrease = margin > 0.0

            grads = torch.autograd.grad(
                losses.sum(),
                delta,
                retain_graph=False,
                create_graph=False,
            )[0]

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                conf = probs[torch.arange(self.B, device=x.device), y]

                s_b = 1.0 + (self.max_groups - 1.0) * (1.0 - conf)
                self.group_budget = s_b.round().long().clamp(
                    min=1,
                    max=self.max_groups,
                )


                self.epsilon[increase] *= (1.0 + rho)
                self.epsilon[decrease] *= (1.0 - rho)
                self.epsilon = torch.clamp(self.epsilon, min=D_MIN)

                new_delta = self._md(
                    g=grads,
                    x=delta,
                    lower=lower,
                    upper=upper,
                    beta=1.0,
                    grid=grid,
                )

                if t == T_warm:
                    self.active_groups = None

                if t >= T_warm:
                    new_delta = self._persistent_group_threshold(new_delta, grid)

                delta.copy_(new_delta)

            LOSSES_.append(losses.detach().cpu())


            x_adv = x + delta


            if self.targeted:
                mask_prediction = torch.argmax(self.model(x_adv), dim=1) == y
                delta_best[mask_prediction] = new_delta[mask_prediction]
            else:
                mask_prediction = torch.argmax(self.model(x_adv), dim=1) != y
                l0s = torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2))
                if t == 0:
                    l02_best = l0s
                else:
                    old_delta = delta_best
                    delta_best[mask_prediction] = new_delta[mask_prediction]
                    mask_l0 = l02_best > l0s
                    l02_best[mask_l0] = l0s[mask_l0]
                    mask_prediction = mask_prediction * mask_l0
                    delta_best[mask_prediction] = old_delta[mask_prediction]
        
        if debug:
            return (x + delta_best).detach(), LOSSES_
        
        return (x + delta_best).detach()

    def _md(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        beta: float,
        grid,
    ) -> torch.Tensor:

        dual_x = torch.log1p(torch.abs(x) / beta) * torch.sign(x)

        max_g2 = torch.amax(g**2, dim=(1, 2, 3))

        if not torch.is_tensor(self.eta):
            self.eta = torch.zeros(
                (g.shape[0], 1, 1, 1),
                device=g.device,
                dtype=g.dtype,
            )

        self.eta = self.eta + max_g2[:, None, None, None]

        eta_t = torch.sqrt(self.eta + 1e-12) / self.learning_rate

        z = dual_x - g / eta_t

        out = []

        for b in range(z.shape[0]):
            #print(f"Batch {b}")
            out.append(
                self._project(
                    z=z[b],
                    l_full=lower[b],
                    u_full=upper[b],
                    D=self.epsilon[b],
                    groups_mask=grid,
                    beta=beta,
                )
            )
            #print(out[-1])

        return torch.stack(out, dim=0)


    def _project(self, z, l_full, u_full, D, groups_mask, beta=1.0):

        """

        Exact sorting-based implementation of Algorithm 1 (GSMD Bregman Projection)

        incorporating the explicit structural forms from Appendix E.

        

        Args:

            z (torch.Tensor): 1D tensor of shape (d,) - The dual variable

            l_full (torch.Tensor): 1D tensor of shape (d,) - Lower bounds

            u_full (torch.Tensor): 1D tensor of shape (d,) - Upper bounds

            D (float/torch.Tensor): The perturbation budget

            groups HxW

            beta (float): Smoothing parameter

            

        Returns:

            torch.Tensor: Projected primal variable x

        """

        device = z.device

        dtype = z.dtype

        original_shape = z.shape

        if torch.is_tensor(D):
            D = D.item()


        #print(z)
        z = z.reshape(-1)
        l_full = l_full.reshape(-1)
        u_full = u_full.reshape(-1)

        full_grid = groups_mask.expand(original_shape).reshape(-1) # C*H*W

        #groups = self.mask_to_list_of_lists(full_grid)

        #print(groups)

        # Algorithm 1 - Line 1: Calculate primal bounds
        c = torch.where(z >= 0, u_full, torch.abs(l_full))
        v = torch.minimum(torch.abs(z), c)
        #print(groups_mask)
        num_groups = int(torch.max(groups_mask).item()) + 1
        #print(num_groups)
        
        groups_list = []
        group_reps = []
        
        # ---------------------------------------------------------
        # Phase 1: Parse the Group Mask natively on GPU
        # ---------------------------------------------------------
        for j in range(num_groups):
            mask_j = (full_grid == j)#(groups_mask == j)

            indices = torch.where(mask_j)[0]
            groups_list.append(indices)
            
            # Find representative i_j 
            v_masked = torch.where(mask_j, v, torch.tensor(-1.0, device=device, dtype=dtype))
            group_reps.append(torch.argmax(v_masked).item())

        # ---------------------------------------------------------
        # Helper: Vectorized Subgradient Evaluator
        # ---------------------------------------------------------
        def get_phi_bounds(j, x_vals_tensor):
            I_j = groups_list[j]
            i_j = group_reps[j]
            
            mask_excl = I_j != i_j
            I_j_excl = I_j[mask_excl]
            
            phi_x = torch.log1p(x_vals_tensor / beta)
            phi_z_ij = math.log1p(abs(z[i_j].item()) / beta)
            
            if I_j_excl.numel() == 0:
                return phi_x - phi_z_ij, phi_x - phi_z_ij
                
            V_group = v[I_j_excl]
            abs_Z_group = torch.abs(z[I_j_excl])
            phi_Z = torch.log1p(abs_Z_group / beta)
            
            # Broadcasting
            mask_minus = x_vals_tensor.unsqueeze(1) <= V_group.unsqueeze(0)
            mask_plus  = x_vals_tensor.unsqueeze(1) < V_group.unsqueeze(0)
            
            count_minus = mask_minus.sum(dim=1)
            count_plus  = mask_plus.sum(dim=1)
            
            sum_phi_z_minus = (mask_minus * phi_Z.unsqueeze(0)).sum(dim=1)
            sum_phi_z_plus  = (mask_plus * phi_Z.unsqueeze(0)).sum(dim=1)
            
            phi_minus_vals = (1 + count_minus) * phi_x - phi_z_ij - sum_phi_z_minus
            phi_plus_vals  = (1 + count_plus)  * phi_x - phi_z_ij - sum_phi_z_plus
            
            return phi_minus_vals, phi_plus_vals

        # ---------------------------------------------------------
        # Phase 2: Primal Sort & Global Breakpoint Generation
        # ---------------------------------------------------------
        G_list = []
        group_thresholds = []
        group_cum_logs = []

        for j in range(num_groups):
            I_j = groups_list[j]
            i_j = group_reps[j]
            
            # 1. Critical Points for G_set (including 0.0)
            mask_excl = I_j != i_j
            I_j_excl = I_j[mask_excl]
            
            C_j_tensor = torch.cat([v[I_j_excl], torch.tensor([0.0], device=device, dtype=dtype)])
            # Sort descending to natively produce ascending dual breakpoints
            unique_C_j_desc = torch.flip(torch.unique(C_j_tensor), dims=[0])
            
            phi_m_C, phi_p_C = get_phi_bounds(j, unique_C_j_desc)
            G_list.append(-phi_m_C)
            G_list.append(-phi_p_C)
            
            # 2. Tracking the active element indices using Primal Sort
            if I_j_excl.numel() > 0:
                #print('HERE', I_j_excl)
                v_excl = v[I_j_excl]
                # Sort the primal points descending
                sorted_v_excl, sort_idx = torch.sort(v_excl, descending=True)
                
                # Evaluate threshold T_i = -Phi'_- (Since v is descending, T_i is natively ASCENDING)
                phi_m_vals, _ = get_phi_bounds(j, sorted_v_excl)
                T_i = -phi_m_vals 
                
                # Record the sorted log indices and precalculate their cumulative sums
                sorted_log_z = torch.log(torch.abs(z[I_j_excl[sort_idx]]) + beta)
                cum_log_z = torch.cumsum(sorted_log_z, dim=0)
                #print("CUMLOGZ",cum_log_z)
            else:
                T_i = torch.empty(0, device=device, dtype=dtype)
                cum_log_z = torch.empty(0, device=device, dtype=dtype)
                
            group_thresholds.append(T_i)
            group_cum_logs.append(cum_log_z)

        # Merge and extract unique sorted dual breakpoints 
        #print(G_list[0])
        G_tensor = torch.cat(G_list)
        G = torch.unique(G_tensor).tolist()
        G.insert(0,0)
        #print(G[0])
        num_intervals = len(G) - 1
        
        c_reps = torch.tensor([c[group_reps[j]].item() for j in range(num_groups)], device=device, dtype=dtype)

        # ---------------------------------------------------------
        # Phase 3: Lazy Interval Binary Search
        # ---------------------------------------------------------
        low = 0
        high = num_intervals - 1
        
        optimal_k = None
        optimal_M = None
        optimal_C = None
        
        while low <= high:
            mid = (low + high) // 2
            lam_hi = G[mid+1]
            lam_lo = G[mid]
            lam_mid_val = (lam_hi + lam_lo) / 2.0
            lam_mid_tensor = torch.tensor(lam_mid_val, device=device, dtype=dtype)
            
            M_mid = torch.zeros(num_groups, device=device, dtype=dtype)
            C_mid = torch.zeros(num_groups, device=device, dtype=dtype)

            #print(lam_mid_val)
            
            for j in range(num_groups):
                # O(log N) Instant check using our pre-sorted thresholds
                m = torch.searchsorted(group_thresholds[j], lam_mid_tensor, side='left').item()
                M_mid[j] = m
                
                if m > 0:
                    # O(1) Instant constant retrieval using pre-recorded cumulative sums
                    sum_log = group_cum_logs[j][m - 1]
                    C_mid[j] = torch.exp(sum_log / m)
                    
            def eval_psi_dynamic(lam):
                valid = M_mid > 0
                M_safe = torch.where(valid, M_mid, torch.ones_like(M_mid))
                
                nabla_phi_star = C_mid * torch.exp(-lam / M_safe) - beta
                nabla_phi_star = torch.where(valid, nabla_phi_star, torch.zeros_like(nabla_phi_star))
                
                rad = torch.clamp(nabla_phi_star, min=0.0)
                rad = torch.minimum(rad, c_reps)
                return torch.sum(rad), rad
            #print(lam_hi,lam_lo)
            psi_hi, _ = eval_psi_dynamic(lam_hi)
            psi_lo, _ = eval_psi_dynamic(lam_lo)
            #print(psi_hi,psi_lo)
            if psi_hi <= D <= psi_lo:
                #print(psi_hi,psi_lo)
                optimal_k = mid
                optimal_M = M_mid
                optimal_C = C_mid
                break
            elif D > psi_lo:
                #low = mid + 1
                high = mid - 1
            else:
                low = mid + 1
                #high = mid - 1

        #print("LH",low,high)

        # ---------------------------------------------------------
        # Phase 4: Inner Bisection & Final Mask Assignment
        # ---------------------------------------------------------
        final_radii = torch.zeros(num_groups, device=device, dtype=dtype)

        #print(final_radii.shape)
        
        if optimal_k is not None:
            #print("optimaL k22", optimal_k)
            lam_hi = G[optimal_k]
            lam_lo = G[optimal_k+1]
            
            def eval_psi_optimal(lam):
                valid = optimal_M > 0
                M_safe = torch.where(valid, optimal_M, torch.ones_like(optimal_M))
                nabla_phi_star = optimal_C * torch.exp(-lam / M_safe) - beta
                nabla_phi_star = torch.where(valid, nabla_phi_star, torch.zeros_like(nabla_phi_star))
                rad = torch.clamp(nabla_phi_star, min=0.0)
                rad = torch.minimum(rad, c_reps)
                return torch.sum(rad)

            for _ in range(30): 
                lam_mid = 0.5 * (lam_lo + lam_hi)
                psi_mid = eval_psi_optimal(lam_mid)
                #print(f"Psi mid: {psi_mid}")
                if psi_mid > D:
                    lam_lo = lam_mid
                else:
                    lam_hi = lam_mid
                    
            # Final pass to get radii
            valid = optimal_M > 0
            M_safe = torch.where(valid, optimal_M, torch.ones_like(optimal_M))
            rad_final = optimal_C * torch.exp(-lam_hi / M_safe) - beta
            rad_final = torch.where(valid, rad_final, torch.zeros_like(rad_final))
            final_radii = torch.minimum(torch.clamp(rad_final, min=0.0), c_reps)
            #print(final_radii.sum())
            #print(f"Radii : {(final_radii!=0).sum()}")
        else:
            # Fallback if D is unreachable
            print('Fallback')
            return z.reshape(original_shape)

        # Instant broadcast of the calculated group radii back to the image
        #print(f"Final radii: {final_radii}")
        limit = final_radii[full_grid]   #final_radii[groups_mask]
        vals = torch.minimum(v, limit)
        #print((vals * torch.sign(z)).reshape(original_shape))
        #print((vals!=0).sum())
        
        #return vals * torch.sign(z)
        return (vals * torch.sign(z)).reshape(original_shape)

    def get_grid_segmentation(self, n):
        block_h = max(1, self.H // n)
        block_w = max(1, self.W // n)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="ij",
        )

        row_idx = torch.div(y_coords, block_h, rounding_mode="floor").clamp(max=n - 1)
        col_idx = torch.div(x_coords, block_w, rounding_mode="floor").clamp(max=n - 1)

        grid_indices = row_idx * n + col_idx
        return grid_indices.long()

    def grid_to_groups(self, grid):
        C, H, W = self.C, self.H, self.W
        num_groups = int(grid.max().item()) + 1

        full_grid = grid.unsqueeze(0).expand(C, H, W).reshape(-1)

        groups = []

        for gid in range(num_groups):
            idx = torch.nonzero(full_grid == gid, as_tuple=False).flatten()
            groups.append(idx)

        return groups

    def _persistent_group_threshold(self, delta, grid):
        B, C, H, W = delta.shape
        num_groups = int(grid.max().item()) + 1
        device = delta.device

        if self.active_groups is None:
            self.active_groups = []

            full_grid = grid.unsqueeze(0).expand(C, H, W).reshape(-1)

            for b in range(B):
                abs_delta = delta[b].abs().reshape(-1)

                scores = torch.zeros(num_groups, device=device, dtype=delta.dtype)

                scores.scatter_reduce_(
                    0,
                    full_grid,
                    abs_delta,
                    reduce="amax",
                )

                s_b = int(self.group_budget[b].item())
                s_b = max(1, min(s_b, num_groups))

                _, active = torch.topk(scores, s_b)
                self.active_groups.append(active)

        out = torch.zeros_like(delta)

        for b in range(B):
            group_mask = torch.zeros(num_groups, device=device, dtype=delta.dtype)
            group_mask[self.active_groups[b]] = 1.0

            pixel_mask = group_mask[grid].unsqueeze(0).expand(C, -1, -1)
            out[b] = delta[b] * pixel_mask

        return out
    

#################################### GSMD-s ##################################

class GSMD_slic(Attack):
    """
    Group-wise sparse attack using mirror descent (GSMD).

    This attack optimizes an adversarial perturbation under
    group-structures using mirror descent.
    """

    def __init__(
        self,
        model,
        *args,
        img_range=(-1, 1),
        targeted: bool = True,
        sequential: bool = False,
        D: float = 1.0,
        l: float = 1.0,
        u: float = 1.0,
        learning_rate = 1.0,
        patches = 15,
        rho= 0.1,
        debug = False,
        **kwargs
    ):
        """
        Initialize the GSMD attack.

        Args:
            model (torch.nn.Module): Target model.
            ver (bool): Verbosity flag.
            img_range (tuple): Allowed pixel range.
            targeted (bool): Whether the attack is targeted.
            sequential (bool): Run attack sequentially per sample.
            D (float): Budget for projection constraint.
            l (float): Lower bound for perturbation.
            u (float): Upper bound for perturbation.
        """
        super().__init__(model, img_range=img_range)
        self.sequential = sequential
        self.D = D
        self.l = l
        self.u = u
        self.targeted = targeted
        self.eta=0.0
        self.learning_rate = learning_rate
        self.deubg = debug
        self.patches = patches
        self.rho = rho

    def __call__(self, x, y, *args, **kwargs):
        """
        Run the attack on a batch of images.

        Depending on the `sequential` flag, the attack is applied
        either per-sample or to the full batch at once.

        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W].
            y (torch.Tensor): Labels of shape [B].

        Returns:
            torch.Tensor: Adversarial examples of same shape as x.
        """
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.perform_att(
                    x_.unsqueeze(0),
                    y_.unsqueeze(0),
                    D = self.D, 
                    l = self.l,
                    u = self.u
                    ).detach()
            return result
        else:
            print('Not sequential')
            return self.perform_att(x, y, D=self.D, l=self.l, u=self.u, rho=self.rho, patches = self.patches, debug=self.deubg)

    def perform_att(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        D: float,
        l: float = None,
        u: float = None,
        steps: int = 200,
        patches=15,
        rho = 0.1,
        debug = False
    ) -> torch.Tensor:
        """
        Perform the attack using mirror descent.

        Args:
            x (torch.Tensor): Input images [B, C, H, W].
            y (torch.Tensor): Ground-truth labels [B].
            D (float): Constraint budget.
            l (float): Lower bound on perturbation.
            u (float): Upper bound on perturbation.
            steps (int): Number of optimization steps.

        Returns:
            torch.Tensor: Adversarial images.
        """
        #self.epsilon = D # 1.0
       
        u = 1-x
        l = -1-x
        self.eta = 0.0 

        x = x.to(self.device)
        y = y.to(self.device)

        

        print(self.device)
        self.B, self.C, self.H, self.W = x.shape
        self.epsilon = torch.full((self.B,), D, device=self.device) # epsilon budget per image

        D_MAX = D * 2.0
        D_MIN = 0.001 
        #rho = 0.1
        print(rho)
        print(patches)
        print(D)

        self.alpha =  0.5 

        segmentation = [torch.tensor(slic(x[i].permute(1,2,0).detach().cpu().numpy(),n_segments=patches**2,start_label=0,sigma=5.0),device=self.device) for i in range(self.B)]#self.get_grid_segmentation(n=patches)
        self.max_groups = torch.tensor([10 for s in segmentation],device=self.device)#torch.tensor([s.max()*0.44 for s in segmentation], device=self.device)
        self.group_budget = self.max_groups
        delta = torch.zeros_like(x, requires_grad=True)

        x_adv = x + delta 
        delta_best = torch.zeros_like(delta)

        T_warm = int(self.alpha * steps)
        LOSSES_ = []
        for t in range(steps):
            logits = self.model(x_adv)
            current_max_labels = torch.argmax(logits,dim=1)

            target_logit = logits[torch.arange(self.B), y] #new
            true_logit = logits[torch.arange(self.B), current_max_labels] # vorher logits[torch.arange(self.B),clean_labels] #new

            if self.targeted:
                losses = F.cross_entropy(logits, y, reduction="none")
                margin = target_logit - true_logit #
                increase = margin < 0.0
                decrease = margin >= 0.0
            else:
                losses = -F.cross_entropy(logits, y, reduction="none")
                margin = true_logit - target_logit
                increase = margin <= 0.0
                decrease = margin > 0.0

            grads = torch.autograd.grad(
                losses.sum(), delta, retain_graph=False, create_graph=False
            )[0]
            with torch.no_grad():

                probs = F.softmax(logits, dim=1)
                conf = probs[torch.arange(self.B), y]
                s_b = 1.0 + (self.max_groups - 1.0) * (1 - conf)
                self.group_budget = s_b.round().long()

                
                self.epsilon[increase] *= (1 + rho)
                self.epsilon[decrease] *= (1 - rho)
                self.epsilon = torch.clamp(self.epsilon, min=D_MIN)

                new_delta = self._md(g=grads,x=delta,lower=l,upper=u,beta=1.0,grid=segmentation)
                
                if t == T_warm:
                    self.active_groups = None 

                if t >= T_warm:
                    new_delta = self._persistent_group_threshold(new_delta,segmentation)

                delta.copy_(new_delta)

            LOSSES_.append(losses.detach().cpu())

            x_adv = x + delta
            if self.targeted:
                mask_prediction = torch.argmax(self.model(x_adv), dim=1) == y
                delta_best[mask_prediction] = new_delta[mask_prediction]
            else:
                mask_prediction = torch.argmax(self.model(x_adv), dim=1) != y
                l0s = torch.norm((x_adv - x).abs().mean(1), p=0, dim=(1,2))
                if t == 0:
                    l02_best = l0s
                else:
                    old_delta = delta_best
                    delta_best[mask_prediction] = new_delta[mask_prediction]
                    mask_l0 = l02_best > l0s
                    l02_best[mask_l0] = l0s[mask_l0]
                    mask_prediction = mask_prediction * mask_l0
                    delta_best[mask_prediction] = old_delta[mask_prediction]

        if debug:
            return (x + delta_best).detach(), LOSSES_
            
          
        return (x + delta_best).detach()


    def _md(self, g: torch.Tensor, x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, beta: float,grid:torch.Tensor) -> torch.Tensor:
        # 1. Map to Dual Space
        dual_x = torch.log(torch.abs(x) / beta + 1.0) * torch.sign(x)
        
        max_g2 = torch.amax(g**2, dim=(1, 2, 3)) 
        self.eta += max_g2[:, None, None, None]
        
        eta_t = torch.sqrt(self.eta) / self.learning_rate
        descent = g / eta_t
        
        # 3. Update in Dual Space
        z = dual_x - descent
        z_sgn = torch.sign(z)
        z_val = torch.abs(z)
        
        # 4. Projection (looping over batch dimension)
        v_list = []
    
        for d in range(dual_x.shape[0]):
            v_list.append(self._project(z_sgn[d], z_val[d], beta, self.epsilon[d], lower[d], upper[d],grid[d]))
        
        return torch.stack(v_list, dim=0)



    def _project(self, y_sgn_full: torch.Tensor, dual_y_val_full: torch.Tensor, beta: float, D: float, l_full: torch.Tensor, u_full: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        c_full = torch.where(y_sgn_full <= 0, torch.abs(l_full), u_full)
        dual_c_full = torch.log(c_full / beta + 1.0)
        
        # target represents min(y, c)
        target = torch.minimum(dual_y_val_full, dual_c_full)
        C, H, W = target.shape
        num_groups = grid.max().item() + 1

        
        flat_target = target.view(C, -1).reshape(-1)
       
        full_grid = grid.expand(C, H, W).reshape(-1)
        #print(full_grid.shape)

        group_winner_val = torch.zeros(num_groups, device=target.device).scatter_reduce(0, full_grid, flat_target, reduce='amax')

        #print(group_winner_val.shape)

        M_ = (grid-grid.min().item()).view(-1) # H*W

        possible_winners = (target == group_winner_val[grid]) 

        flat_possible = possible_winners.view(-1)
        final_indices = torch.zeros(num_groups, dtype=torch.long, device=target.device)
        final_indices = final_indices.scatter(0, full_grid[flat_possible], torch.arange(C*H*W, device=target.device)[flat_possible])

        # 6. Extract the representatives for Binary Search
        dual_y_val = dual_y_val_full.view(-1)[final_indices]
        dual_c = dual_c_full.view(-1)[final_indices]
        c = c_full.view(-1)[final_indices]
        


        # Check if budget D is already satisfied
        phi_0 = beta * torch.exp(torch.clamp(torch.minimum(dual_y_val, dual_c), min=0.0)) - beta
        if torch.sum(phi_0) <= D:
            v = phi_0
        else:
            # Binary Search for Lagrange Multiplier
            z = torch.cat(((dual_y_val).reshape(-1), (dual_y_val - dual_c).reshape(-1)))
            z = torch.sort(z[z >= 0])[0]
            
            idx_l, idx_u = 0, len(z) - 1
            while idx_u - idx_l > 1:
                idx = (idx_u + idx_l) // 2
                lam = z[idx]
                phi = torch.sum(beta * torch.exp(torch.clamp(torch.minimum(dual_y_val - lam, dual_c), min=0.0)) - beta)
                if phi > D:
                    idx_l = idx
                else:
                    idx_u = idx
                    
            lam_lower, lam_upper = z[idx_u], z[idx_l]
            
            
            lam = (lam_lower + lam_upper) / 2.0
            idx_clip = (dual_y_val - lam) >= dual_c
            idx_active = ((dual_y_val - lam) < dual_c) & ((dual_y_val - lam) > 0)
            
            v = torch.where(idx_clip, c, torch.zeros_like(c))
            num_active = torch.sum(idx_active)
            if num_active != 0:
                sum_active = D - torch.sum(c[idx_clip])
                normaliser = (sum_active + beta * num_active) / torch.sum(beta * torch.exp(dual_y_val)[idx_active])
                v = torch.where(idx_active, beta * torch.exp(dual_y_val) * normaliser - beta, v)

        _y_val_full = beta * torch.exp(torch.minimum(dual_y_val_full, dual_c_full)) - beta

        v = v[M_].view(self.H,self.W).expand(self.C,-1,-1)

        v_full = torch.minimum(_y_val_full, v)
        return v_full * y_sgn_full


    def _persistent_group_threshold(self, delta, grid):
        B, C, H, W = delta.shape
        #num_groups = grid[0].max().item() + 1
        device = delta.device

        if self.active_groups is None:
            self.active_groups = []

            for b in range(B):
                num_groups = grid[b].max().item() + 1
                abs_delta = delta[b].abs().reshape(-1)
                full_grid = grid[b].unsqueeze(0).expand(C, H, W).reshape(-1)
                #full_grid = grid.unsqueeze(0).expand(C, H, W).reshape(-1)

                scores = torch.zeros(num_groups, device=device)
                scores.scatter_reduce_(0, full_grid, abs_delta, reduce="amax")

                s_b = int(self.group_budget[b].item())
                _, active = torch.topk(scores, s_b)

                self.active_groups.append(active)

        out = torch.zeros_like(delta)

        for b in range(B):
            num_groups = grid[b].max().item() + 1
            group_mask = torch.zeros(num_groups, device=device)
            group_mask[self.active_groups[b]] = 1.0

            pixel_mask = group_mask[grid[b]].unsqueeze(0).expand(C, -1, -1)           
            out[b] = delta[b] * pixel_mask

        return out


##################################### GSE #####################################

class GSEAttack(Attack):
    def __init__(self, model, *args, ver=False, img_range=(-1, 1), search_steps=10,
                 targeted=False, sequential=False, search_factor=2,
                 gb_size=5, sgm=1.5, mu=1, sigma=0.0025, iters=200, k_hat=10,
                 q=0.25, **kwargs):
        '''
        Implementation of the GSE attack.

        args:
        model:         Callable, PyTorch classifier.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        search_steps:  Int, number of steps for line search on the trade-off
                       parameter.
        targeted:      Bool, given label is used as a target label if True.
        sequential:    Bool, perturbations are computed sequentially for all
                       images in the batch if True. For fair comparison to
                       Homotopy attack.
        search_factor: Float, factor to increase/decrease the trade-off
                       parameter until an upper/lower bound for the line search
                       is found.
        gb_size:       Odd int, size of the Gaussian blur kernel.
        sgm:           Float, sigma of the gaussian blur kernel
        mu:            Float, trade-off parameter for 2-norm regularization.
        sigma:         Float, step size
        iters:         Int, number of iterations.
        k_hat:         Int, number of iterations before transitioning to NAG.
        q:             Float, inverse of increase factor for adjust_lambda.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.ver = ver
        self.search_steps = search_steps
        self.sequential = sequential
        self.search_factor = search_factor
        self.gb_size = gb_size
        self.sgm = sgm
        self.mu = mu
        self.sigma = sigma
        self.iters = iters
        self.k_hat = k_hat
        self.q = q


    def adjust_lambda(self, lam, noise):
        '''
        Adjust trade-off parameters (lambda) to update search space.
        '''
        x = noise.detach().clone().abs().mean(dim=1, keepdim=True).sign()
        gb = torchvision.transforms.GaussianBlur((self.gb_size, self.gb_size),
                                                 sigma=self.sgm)
        x = gb(x) + 1
        x = torch.where(x == 1, self.q, x)
        lam /= x[:, 0, :, :]
        return lam


    def section_search(self, x, y, steps=50):
        '''
        Section search for finding the maximal lambda such that the
        perturbation is non-zero after the first iteration.
        '''
        noise = torch.zeros_like(x, requires_grad=True)
        loss = (self.f(x + noise, y).sum() + self.mu
                * torch.norm(noise, p=2, dim=(1,2,3)).sum())
        loss.backward()
        grad = noise.grad
        noise.detach_()
        ones = torch.ones_like(x)[:, 0, :, :]

        # define upper and lower bound for line search
        lb = torch.zeros((y.size(0),), dtype=torch.float,
                         device=self.device).view(-1, 1, 1)
        ub = lb.clone() + 0.001
        mask = torch.norm(self.prox(grad.clone() * self.sigma,
                                      ones * ub * self.sigma),
                          p=0, dim=(1,2,3)) != 0
        while mask.any():
            ub[mask] *= 2
            mask = torch.norm(self.prox(grad.clone() * self.sigma,
                                          ones * ub * self.sigma),
                              p=0, dim=(1,2,3)) != 0

        # perform search
        for _ in range(steps):
            cur = (ub + lb) / 2
            mask = torch.norm(self.prox(grad.clone() * self.sigma,
                                          ones * cur * self.sigma),
                              p=0, dim=(1,2,3)) == 0
            ub[mask] = cur[mask]
            mask = torch.logical_not(mask)
            lb[mask] = cur[mask]
        cur = (lb + ub).view(-1) / 2
        return 0.01 * cur


    def __call__(self, x, y, *args, **kwargs):
        '''
        Call the attack for a batch of images x or sequentially for all images
        in x depending on self.sequential.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.perform_att(x_.unsqueeze(0),
                                             y_.unsqueeze(0),
                                             mu=self.mu,
                                             k_hat=self.k_hat).detach()
            return result
        else:
            return self.perform_att(x, y, mu=self.mu,
                                    k_hat=self.k_hat)


    def perform_att(self, x, y, mu, k_hat):
        '''
        Perform GSE attack on a batch of images x with corresponding labels y.
        '''
        x = x.to(self.device)
        y = y.to(self.device)
        B, C, _, _ = x.shape
        lams = self.section_search(x, y)
        # save x, y, and lams for resetting them at the beginning of every
        # section search step
        save_x = x.clone()
        save_y = y.clone()
        save_lams = lams.clone()
        # upper and lower bounds for section learch
        ub_lams = torch.full_like(lams, torch.inf)
        lb_lams = torch.full_like(lams, 0.0)
        # tensor for saving succesful adversarial examples in inner loop
        result = x.clone()
        # tensor for saving best adversarial example so far
        result2 = x.clone()
        best_l0 = torch.full((B,), torch.inf, device=self.device).type(x.type())

        # section search
        for step in range(self.search_steps):
            x = save_x.clone()
            y = save_y.clone()
            lams = save_lams.clone()
            lam = torch.ones_like(x)[:, 0, :, :] * lams.view(-1, 1, 1)
            sigma = torch.full_like(lams, self.sigma).view(-1, 1, 1, 1)
            # increase the step size if were not successful with only 2 steps left
            sigma[((result - x).view(B, -1).abs().sum(-1) == 0) & (step >= self.search_steps-2)] *= 10
            # tensor for tracking for which images adv. examples have been found
            active = torch.ones(B, dtype=bool, device=self.device)
            # set initial perturbation to zero
            noise = torch.zeros_like(x, requires_grad = True)
            noise_old = noise.clone()
            lr = 1

            # attack
            for j in range(self.iters):
                if self.ver:
                    print(f'\rSearch step {step + 1}/{self.search_steps}, ' +
                          f'Prox.Grad. Iteration {j + 1}/{self.iters}, ' +
                          f'Images left: {x.shape[0]}', end='')
                if len(x) == 0:
                    break

                self.model.zero_grad()
                loss = (self.f(x + noise, y).sum() + mu
                        * (torch.norm(noise, p=2, dim=(1,2,3)) ** 2).sum())
                loss.backward()

                with torch.no_grad():
                    lr_ = (1 + math.sqrt(1 + 4 * lr**2)) / 2
                    if j == k_hat:
                        lammask = (lam > lams.view(-1, 1, 1))[:, None, :, :]
                        lammask = lammask.repeat(1, C, 1, 1)
                        noise[lammask] = 0
                        noise_old = noise.clone()
                    if j < k_hat:
                        noise = noise - sigma * noise.grad.data
                        noise = self.prox(noise, lam * sigma.view(-1, 1, 1))
                        noise_tmp = noise.clone()
                        noise = lr / lr_ * noise + (1 - (lr/ lr_)) * noise_old
                        noise_old = noise_tmp.clone()
                        lam = self.adjust_lambda(lam, noise)
                    else:
                        noise = noise - sigma * noise.grad.data
                        noise[lammask] = 0
                        noise_tmp = noise.clone()
                        noise = lr / lr_ * noise + (1 - (lr/ lr_)) * noise_old
                        noise_old = noise_tmp.clone()

                    # clamp adv. example to valid range
                    x_adv = torch.clamp(x + noise, *self.img_range)
                    noise = x_adv - x
                    lr = lr_

                    preds = torch.argmax(self.model(x_adv), dim=1)
                    mask = preds == y if self.targeted else preds != y
                    # save successful examples
                    if mask.any():
                        tmp = result[active]
                        tmp[mask] = x_adv[mask]
                        result[active] = tmp
                        mask = torch.logical_not(mask)
                        active[active.clone()] = mask
                        x, y, noise = x[mask], y[mask], noise[mask]
                        lams, lam = lams[mask], lam[mask]
                        noise_old = noise_old[mask]
                        sigma = sigma[mask]
                        if j >= k_hat:
                            lammask = lammask[mask]

                noise.requires_grad = True

            # section search
            # no adv. example found => decrease upper bound and current lambda
            # adv. example found => save it if the "0-norm" is better than of the
            # previous adv. example, increase lower bound and current lambda
            for i in range(B):
                if active[i]:
                    ub_lams[i] = save_lams[i]
                    save_lams[i] = 0.9 * lb_lams[i] + 0.1 * save_lams[i]
                else:
                    l0 = self.l20((result[i] - save_x[i]).unsqueeze(0)).to(self.device)
                    if l0 < best_l0[i]:
                        best_l0[i] = l0
                        result2[i] = result[i].clone()
                    if torch.isinf(ub_lams[i]):
                        lb_lams[i] = save_lams[i]
                        save_lams[i] *= self.search_factor
                    else:
                        lb_lams[i] = save_lams[i]
                        save_lams[i] = (ub_lams[i] + save_lams[i]) / 2

        if self.ver:
            print('')
        return result2.detach()


    def f(self, x, y, kappa=0):
        '''
        CW loss function
        '''
        logits = self.model(x)
        one_hot_y = F.one_hot(y, logits.size(1))
        Z_t = torch.sum(logits * one_hot_y, dim=1)
        Z_i = torch.amax(logits * (1 - one_hot_y) - (one_hot_y * 1e5), dim=1)
        if self.targeted:
            return F.relu(Z_i - Z_t + kappa)
        else:
            return F.relu(Z_t - Z_i + kappa)


    def extract_patches(self, x):
        '''
        Extracts and returns all overlapping size by size patches from
        the image batch x.
        '''
        B, C, _, _ = x.shape
        size = 8
        kernel = torch.zeros((size ** 2, size ** 2))
        kernel[range(size**2), range(size**2)] = 1.0
        kernel = kernel.view(size**2, 1, size, size)
        kernel = kernel.repeat(C, 1, 1, 1).to(x.device)
        out = F.conv2d(x, kernel, groups=C)
        out = out.view(B, C, size, size, -1)
        out = out.permute(0, 4, 1, 2, 3)
        return out.contiguous()
    
    def l20(self, x):
        '''
        Computes d_{2,0}(x[i]) for all perturbations x[i] in the batch x
        as described in section 3.2.
        '''
        l20s = []
        for x_ in x:
            patches = self.extract_patches(x_.unsqueeze(0))
            l2s = torch.norm(patches, p=2, dim=(2, 3, 4))
            l20s.append((l2s != 0).float().sum().item())
        return torch.tensor(l20s)


    def prox(self, grad_loss_noise, lam):
        '''
        Computes the proximal operator of the 1/2-norm of the gradient of the
        adversarial loss wrt current noise.
        '''

        lam = lam[:, None, :, :]
        sh = list(grad_loss_noise.shape)
        lam = lam.expand(*sh)

        p_lam = (54 ** (1 / 3) / 4) * lam ** (2 / 3)

        mask1 = (grad_loss_noise > p_lam)
        mask2 = (torch.abs(grad_loss_noise) <= p_lam)
        mask3 = (grad_loss_noise < -p_lam)
        mask4 = mask1 + mask3

        phi_lam_x = torch.arccos((lam / 8) * (torch.abs(grad_loss_noise) / 3)
                                 ** (-1.5))

        grad_loss_noise[mask4] = ((2 / 3) * torch.abs(grad_loss_noise[mask4])
                                  * (1 + torch.cos((2 * math.pi) / 3
                                  - (2 * phi_lam_x[mask4]) / 3)))
        grad_loss_noise[mask3] = -grad_loss_noise[mask3]
        grad_loss_noise[mask2] = 0

        return grad_loss_noise



################################### Fwnucl ####################################

class FWnucl(Attack):
    def __init__(self, model, *args, iters=200, img_range=(-1, 1), ver=False,
                 targeted=False, eps=5, **kwargs):
        '''
        Implementation of the nuclear group norm attack.

        args:
        model:         Callable, PyTorch classifier.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        eps:           Float, radius of the nuclear group norm ball.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.iters = iters
        self.ver = ver
        self.eps = eps
        self.gr = (math.sqrt(5) + 1) / 2


    def __loss_fn(self, x, y, lossfn):
        '''
        Compute loss depending on self.targeted.
        '''
        if self.targeted:
            return -lossfn(x, y)
        else:
            return lossfn(x, y)


    def __call__(self, x, y, *args, **kwargs):
        '''
        Perform the nuclear group norm attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        
        x = x.to(self.device)
        y = y.to(self.device)
        lossfn = nn.CrossEntropyLoss()
        noise = torch.zeros_like(x)
        noise.requires_grad = True

        for t in range(self.iters):
            if self.ver:
                print(f'\rIteration {t+1}/{self.iters}', end='')
            self.model.zero_grad()
            out = self.model(x + noise)
            loss = self.__loss_fn(out, y, lossfn)
            loss.backward()
            s = self.__groupNuclearLMO(noise.grad.data, eps=self.eps)
            with torch.no_grad():
                gamma = self.__lineSearch(x, s, noise, y)
                noise = (1 - gamma) * noise + gamma * s
            noise.requires_grad = True

        x = torch.clamp(x + noise, -1, 1)
        if self.ver:
            print("")
        return x.detach()


    def __lineSearch(self, x, s, noise, y, steps=25):
        '''
        Perform line search for the step size.
        '''
        a = torch.zeros(y.shape, device=self.device).view(-1, 1, 1, 1)
        b = torch.ones(y.shape, device=self.device).view(-1, 1, 1, 1)
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr
        lossfn = nn.CrossEntropyLoss(reduction='none')
        sx = s - noise

        for i in range(steps):
            loss1 = self.__loss_fn(self.model(x + noise + c * sx), y, lossfn)
            loss2 = self.__loss_fn(self.model(x + noise + d * sx), y, lossfn)
            mask = loss1 > loss2

            b[mask] = d[mask]
            mask = torch.logical_not(mask)
            a[mask] = c[mask]

            c = b - (b - a) / self.gr
            d = a + (b - a) / self.gr

        return (b + a) / 2


    def __groupNuclearLMO(self, x, eps=5):
        '''
        LMO for the nuclear group norm ball.
        '''

        B, C, H, W = x.shape
        size = 32 if H > 64 else 4

        # turn batch of images into batch of size by size pixel groups per
        # color channel
        xrgb = [x[:, c, :, :] for c in range(C)]
        xrgb = [xc.unfold(1, size, size).unfold(2, size, size) for xc in xrgb]
        xrgb = [xc.reshape(-1, size, size) for xc in xrgb]

        # compute nuclear norm of each patch (sum norms over color channels)
        norms = torch.linalg.svdvals(xrgb[0])
        for xc in xrgb[1:]:
            norms += torch.linalg.svdvals(xc)
        norms = norms.sum(-1).reshape(B, -1)

        # only keep the patch g* with the largest nuclear norm for each image
        idxs = norms.argmax(dim=1).view(-1, 1)
        xrgb = [xc.reshape(B, -1, size, size) for xc in xrgb]
        xrgb = [xc[torch.arange(B).view(-1, 1), idxs].view(B, size, size)
                for xc in xrgb]

        # build index tensor corr. to the position of the kept patches in x
        off = (idxs % (W / size)).long() * size
        off += torch.floor(idxs / (W / size)).long() * W * size
        idxs = torch.arange(0, size**2,
                            device=self.device).view(1, -1).repeat(B, 1) + off
        off = torch.arange(0, size,
                           device=self.device).view(-1, 1).repeat(1, size)
        off = off * W  - off * size
        idxs += off.view(1, -1)

        # compute singular vector pairs corresponding to largest singular value
        # and final perturbation (LMO solution)
        pert = torch.zeros_like(x)
        for i, xc in enumerate(xrgb):
            U, _, V = torch.linalg.svd(xc)
            U = U[:, :, 0].view(B, size, 1)
            V = V.transpose(-2, -1)[:, :, 0].view(B, size, 1)
            pert_gr = torch.bmm(U, V.transpose(-2, -1)).reshape(B, size * size)
            idx = torch.arange(B).view(-1, 1)
            pert_tmp = pert[:, i, :, :].view(B, -1)
            pert_tmp[idx, idxs] = pert_gr * eps
            pert[:, i, :, :] = pert_tmp.view(B, H, W)

        return pert



################################## StrAttack ##################################

class StrAttack(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1), kappa=0,
                 max_iter=200, ver=False, search_steps=8, max_c=1e10, rho=1,
                 c=2.5, retrain=True, **kwargs):
        '''
        Implementation of StrAttack: https://arxiv.org/abs/1808.01664
        Adapted from https://github.com/KaidiXu/StrAttack

        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        kappa:         Float, confidence parameter for CW loss.
        max_iter:      Int, number of iterations.
        ver:           Bool, print progress if True.
        search_steps:  Int, number of binary search steps.
        max_c:         Float, upper bound for regularizaion parameter.
        rho:           Float, ADMM parameter.
        c:             Float, initial regularization parameter.
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.kappa = kappa
        self.max_iter = max_iter
        self.ver = ver
        self.search_steps = search_steps
        self.max_c = max_c
        self.rho = rho
        self.c = c
        self.retrain = retrain

    def __compare(self, x, y):
        return torch.equal(x, y) if self.targeted else not torch.equal(x, y)

    def __f(self, x, y):
        '''
        CW loss function
        '''
        logits = self.model(x)
        one_hot_labels = F.one_hot(y, logits.size(1)).to(self.device)
        Z_i = torch.max(logits * (1 - one_hot_labels) - one_hot_labels
                        * 10000., dim=1)[0]
        Z_t = torch.sum(logits * one_hot_labels, dim=1)
        Zdif = Z_i - Z_t if self.targeted else Z_t - Z_i
        return torch.clamp(Zdif + self.kappa, min=0.0)


    def __call__(self, imgs, labs, *args, **kwargs):
        '''
        Perform StrAttack on a batch of images x with corresponding labels y.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        c_ = self.c
        imgs = imgs.to(self.device)
        labs = labs.to(self.device)
        sh = imgs.shape
        batch_size = sh[0]

        alpha, tau, gamma = 5, 2, 1
        eps = torch.full_like(imgs, 1.0)
        # 16 for imagenet, 2 for CIFAR and MNIST
        filterSize = 8 if sh[-1] > 32 else 2
        stride = filterSize
        # convolution kernel used to compute norm of each group
        slidingM = torch.ones((1, sh[1], filterSize, filterSize), device=self.device)

        cs = torch.ones(batch_size, device=self.device) * c_
        lower_bound = torch.zeros(batch_size)
        upper_bound = torch.ones(batch_size) * self.max_c

        o_bestl2 = torch.full_like(labs, 1e10, dtype=torch.float)
        o_bestscore = torch.full_like(labs, -1, dtype=torch.float)
        o_bestattack = imgs.clone()
        o_besty = torch.ones_like(imgs)

        for step in range(self.search_steps):

            bestl2 = torch.full_like(labs, 1e10, dtype=torch.float)
            bestscore = torch.full_like(labs, -1, dtype=torch.float)

            z, v, u, s = (torch.zeros_like(imgs) for _ in range(4))

            for iter_ in range(self.max_iter):
                if (not iter_%10 or iter_ == self.max_iter - 1) and self.ver:
                    print(f'\rIteration: {iter_+1}/{self.max_iter}, ' +
                          f'Search Step: {step+1}/{self.search_steps}', end='')

                # first update step (7) / Proposition 1
                delta = self.rho / (self.rho + 2 * gamma) * (z - u / self.rho)

                b = z - s / self.rho
                tmp = torch.minimum(self.img_range[1] - imgs, eps)
                w = torch.where(b > tmp, tmp, b)
                tmp = torch.maximum(self.img_range[0] - imgs, -eps)
                w = torch.where(b < tmp, tmp, w)

                c = z - v / self.rho
                cNorm = torch.sqrt(F.conv2d(c ** 2, slidingM, stride=stride))
                cNorm = torch.where(cNorm == 0, torch.full_like(cNorm, 1e-12), cNorm)
                cNorm = F.interpolate(cNorm, scale_factor=filterSize)
                y = torch.clamp((1 - tau / (self.rho * cNorm)), 0) * c

                # second update step (8) / equation (15)
                z_grads = self.__get_z_grad(imgs, labs, z.clone(), cs)
                eta = alpha * math.sqrt(iter_ + 1)
                coeff = (1 / (eta + 3 * self.rho))
                z = coeff * (eta * z + self.rho * (delta + w + y) + u + s + v - z_grads)

                # third update step (9)
                u = u + self.rho * (delta - z)
                v = v + self.rho * (y - z)
                s = s + self.rho * (w - z)

                # get info for binary search
                x = imgs + y
                scores = self.model(x)
                l2s = torch.sum((z ** 2).reshape(z.size(0), -1), dim=-1)

                for i, (l2, sc, x_) in enumerate(zip(l2s, scores, x)):
                    if l2 < bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i]):
                        bestl2[i] = l2
                        bestscore[i] = asc
                    if l2 < o_bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i]):
                        o_bestl2[i] = l2
                        o_bestscore[i] = asc
                        o_bestattack[i] = x_.detach().clone()
                        o_besty[i] = y[i]

            for i in range(batch_size):
                if (self.__compare(bestscore[i], labs[i]) and bestscore[i] != -1 and bestl2[i] == o_bestl2[i]):
                    upper_bound[i] = min(upper_bound[i], cs[i])
                    if upper_bound[i] < 1e9:
                        cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], cs[i])
                    if upper_bound[i] < 1e9:
                        cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        cs[i] *= 5

        del v, u, s, z_grads, w, tmp

        if self.retrain:
            cs = torch.full_like(labs, 5.0, dtype=torch.float)
            zeros = torch.zeros_like(imgs)

            for step in range(8):
                bestl2 = torch.full_like(labs, 1e10, dtype=torch.float)
                bestscore = torch.full_like(labs, -1, dtype=torch.float)

                Nz = o_besty[o_besty != 0]
                e0 = torch.quantile(Nz.abs(), 0.03)
                A2 = torch.where(o_besty.abs() <= e0, 0, 1)
                z1 = o_besty
                u1 = torch.zeros_like(imgs)
                tmpc = self.rho / (self.rho + gamma / 100)

                for j in range(200):
                    if self.ver and not j % 10:
                        print(f'\rRetrain iteration: {step+1}/8, ' +
                              f'Search Step: {j+1}/200', end='')

                    tmpA = (z1 - u1) * tmpc
                    tmpA1 = torch.where(o_besty.abs() <= e0, zeros, tmpA)
                    cond = torch.logical_and(tmpA >
                                             torch.minimum(self.img_range[1] - imgs, eps),
                                             o_besty.abs() > e0)
                    tmpA2 = torch.where(cond, torch.minimum(self.img_range[1] - imgs, eps),
                                        tmpA1)
                    cond = torch.logical_and(tmpA <
                                             torch.maximum(self.img_range[0] - imgs, -eps),
                                             o_besty.abs() > e0)
                    deltA = torch.where(cond, torch.maximum(self.img_range[0] - imgs, -eps),
                                        tmpA2)

                    x = imgs + deltA
                    scores = self.model(x)
                    l2s = torch.sum((z ** 2).reshape(z.size(0), -1), dim=-1)
                    grad = self.__get_z_grad(imgs, labs, deltA, cs)

                    stepsize = 1 / (alpha + 2 * self.rho)
                    z1 = stepsize * (alpha * z1 * self.rho
                                     * (deltA + u1) - grad * A2)
                    u1 = u1 + deltA - z1

                    for i, (l2, sc, x_) in enumerate(zip(l2s, scores, x)):
                        if (l2 < bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i])):
                            bestl2[i] = l2
                            bestscore[i] = asc
                        if (l2 < o_bestl2[i] and self.__compare(asc:=torch.argmax(sc), labs[i])):
                            o_bestl2[i] = l2
                            o_bestscore[i] = asc
                            o_bestattack[i] = x_.detach().clone()
                            o_besty[i] = deltA[i]


                for i in range(batch_size):
                    if self.__compare(bestscore[i], labs[i]) and bestscore[i] != -1 and bestl2[i] == o_bestl2[i]:
                        upper_bound[i] = min(upper_bound[i], cs[i])
                        if upper_bound[i] < 1e9:
                            cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        lower_bound[i] = max(lower_bound[i], cs[i])
                        if upper_bound[i] < 1e9:
                            cs[i] = (lower_bound[i] + upper_bound[i]) / 2
                        else:
                            cs[i] *= 5

        if self.ver:
            print('')
        return o_bestattack


    def __get_z_grad(self, imgs, y, z, cs):
        '''
        Compute and return gradient of loss wrt. z.
        '''
        z.requires_grad = True
        tmp = self.__f(z + imgs, y)
        loss = torch.mean(cs * tmp)
        loss.backward()
        z.detach_()
        return z.grad.data



################################# Homotopy ####################################

class HomotopyAttack(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1), ver=False,
                 loss_type='cw', max_epsilon=0.1, dec_factor=0.98, val_c=1e-2,
                 val_w1=1e-1, val_w2=1e-3, max_update=1, maxiter=100,
                 val_gamma=0.8, eta=0.9, delta=0.3, rho=0.8, beta=1e-2,
                 iter_init=50, kappa=0.0, iter_inc=[], n_segments=500, **kwargs):
        '''
        Implementation of group-wise sparse Homotopy attack:
        https://arxiv.org/abs/2106.06027
        Adapted from https://github.com/VITA-Group/SparseADV_Homotopy

        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        ver:           Bool, print progress if True.
        loss_type:     Str 'cw' or 'ce', CW loss or corss entropy loss
        max_epsilon:   Float, upper bound for perturbation magnitude.
        def_factor:    Float, decrease factor for lambda
        val_c:         Float, factor for lambda after initial search.
        val_w1:        Float, trade-off parameter for classification loss.
        val_w2:        Float, trade-off parameter for regularization.
        max_update:    Int, maximum number of super pixels zo be updated per
                       iteration.
        maxiter:       Int, maximum number of nmAPG iterations.
        val_gamma:     Float, constant for criterion (Eq. 12)
        eta:           Float, nmAPG parameter.
        delta:         Float, nmAPG parameter.
        rho:           Float, nmAPG parameter.
        beta:          Float, parameter for increasing lambda during the first
                       search.
        iters_init:    Int, used for computing the number of iterations for the
                       ater attack routine.
        kappa:         Float, confidence parameter for the CW loss.
        iter_inc:      List of ints, used for computing the number of
                       iterations for the ater attack routine.
        n_segments:    Int, initial number of super pixels given to SLIC.
        '''

        super().__init__(model, targeted=targeted, img_range=img_range)
        self.ver = ver
        self.loss_type = loss_type
        self.max_epsilon = max_epsilon
        self.dec_factor = dec_factor
        self.val_c = val_c
        self.val_w1 = val_w1
        self.val_w2 = val_w2
        self.max_update = max_update
        self.maxiter = maxiter
        self.val_gamma = val_gamma
        self.eta = eta
        self.delta = delta
        self.rho = rho
        self.beta = beta
        self.iter_init = iter_init
        self.iter_inc = iter_inc
        self.kappa = kappa
        self.n_segments = n_segments


    def __call__(self, x, y, *args, **kwargs):
        '''
        Performs the attack.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        result = x.clone().cpu()
        for i, (x_, y_) in enumerate(zip(x, y)):
            if self.ver:
                print(f'Image {i+1}/{x.shape[0]}')
            x_ = x_.unsqueeze(0).to(self.device)
            y_ = y_.unsqueeze(0).to(self.device)
            result[i] += self.homotopy(x_, y_).cpu()[0]

        return result.detach()


    def after_attack(self, x, original_img, target_class, post, iters):

        if post == 1:
            s1 = 1e-3
            s2 = 1e-4
            max_iter = 40000
        else:
            s1 = self.val_w2
            s2 = self.val_w1
            max_iter = iters

        mask = torch.where(torch.abs(x.data) > 0, torch.ones(1).to(self.device),
                           torch.zeros(1).to(self.device))
        pre_x = x.data

        for _ in tqdm(range(max_iter)):

            temp = Variable(x.data, requires_grad=True)
            logist = self.model(temp + original_img.data)
            if self.targeted:
                if self.loss_type == 'ce':
                    ce = torch.nn.CrossEntropyLoss()
                    Loss = ce(logist,torch.ones(1).long().to(self.device)*target_class)
                elif self.loss_type == 'cw':
                    Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
            else:
                Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)

            self.model.zero_grad()
            if temp.grad is not None:
                temp.grad.data.fill_(0)
            Loss.backward()
            grad = temp.grad


            temp2 = Variable(x.data, requires_grad=True)
            Loss2 = torch.norm(temp2, p=float("inf"))
            self.model.zero_grad()
            if temp2.grad is not None:
                temp2.grad.data.fill_(0)
            Loss2.backward()
            grad2 = temp2.grad

            pre_x = x.data
            if post == 0:
                temp2 = temp2.data - s1*grad2.data*mask - s2*grad.data*mask
            else:
                temp2 = temp2.data - s1*grad2.data*mask

            thres = self.max_epsilon
            temp2 = torch.clamp(temp2.data, -thres, thres)
            temp2 = torch.clamp(original_img.data+temp2.data, *self.img_range)

            x = temp2.data - original_img.data


            logist = self.model(x.data + original_img.data)
            _,pre=torch.max(logist,1)
            if(post == 1):
                if self.targeted:
                    if(pre.item() != target_class):
                        return pre_x
                else:
                    if(pre.item() == target_class):
                        return pre_x

        return x


    def F(self, x, lambda1, original_img, target_class):
        temp = Variable(x.data, requires_grad=False)
        logist = self.model(temp+original_img.data)
        if self.targeted:
            if self.loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss = ce(logist,torch.ones(1).long().to(self.device)*target_class)
            elif self.loss_type == 'cw':
                Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        else:
            Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        res = Loss.item() + lambda1*torch.norm(x.data,0).item()
        self.model.zero_grad()
        return res


    # def prox_pixel(self, x, alpha, lambda1, original_img):
    #     '''
    #     Applies the proximal operator of the group norm to x. Each group
    #     corresponds to a superpixel in the original image.
    #     '''
    #     B, C, H, W = x.shape
    #     temp_x = x.data * torch.ones_like(x)

    #     thres = self.max_epsilon
    #     clamp_x = torch.clamp(temp_x, -thres, thres)

    #     temp_img = original_img + clamp_x
    #     temp_img = torch.clamp(temp_img, *self.img_range)
    #     clamp_x = temp_img.data - original_img.data

    #     temp_x_norm = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
    #     pi_x_norm = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
    #     for i in range(self.groups.max()):
    #         mask = self.groups == i + 1
    #         mask = mask[None, :, :].repeat(1, C, 1, 1)
    #         temp_x_norm[i] = (temp_x[mask] ** 2).sum()
    #         pi_x_norm[i] = (clamp_x[mask] ** 2).sum()

    #     val = 1 / (2 * alpha * lambda1)
    #     cond = 1 + val * pi_x_norm > val * temp_x_norm
    #     idxs = cond.float().nonzero()
    #     res = x.clone()
    #     for i in idxs:
    #         mask = (self.groups == i.item() + 1)
    #         mask = mask[None, :, :].repeat(1, C, 1, 1)
    #         res[mask] *= 0
    #     return res

    def prox_pixel(self, x, alpha, lambda1, original_img):
        thres = self.max_epsilon
        # 1. Project x into the epsilon ball and image range
        clamp_x = torch.clamp(x.data, -thres, thres)
        temp_img = torch.clamp(original_img + clamp_x, *self.img_range)
        clamp_x = temp_img - original_img

        # 2. Vectorized group norm calculation
        # Instead of looping, we use the groups as indices for scatter_add
        C = x.shape[1]
        # Flatten spatial dims: [1, C, H*W]
        flat_x = x.view(1, C, -1)
        flat_clamp = clamp_x.view(1, C, -1)
        flat_groups = self.groups.view(-1) - 1 # 0-indexed

        # Calculate squared norms per group across all channels
        # sum(x^2) per group
        x_sq = (flat_x**2).sum(dim=1) 
        clamp_sq = (flat_clamp**2).sum(dim=1)
        
        num_groups = self.groups.max().item()
        group_x_norm = torch.zeros(num_groups, device=x.device).scatter_add_(0, flat_groups, x_sq.squeeze())
        group_clamp_norm = torch.zeros(num_groups, device=x.device).scatter_add_(0, flat_groups, clamp_sq.squeeze())

        # 3. Apply the Proximal condition
        val = 1 / (2 * alpha * lambda1)
        # Boolean mask of which groups to KEEP (not zero out)
        keep_mask_groups = (1 + val * group_clamp_norm <= val * group_x_norm)
        
        # Map group decisions back to pixels
        keep_mask_pixels = keep_mask_groups[flat_groups].view(1, 1, *self.groups.shape)
        
        return x * keep_mask_pixels



    def pert_groups(self, x):
        '''
        Checks which superpixels have non-zero perturbation.
        '''
        C = x.shape[1]
        pert = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
        for i in range(self.groups.max()):
            mask = self.groups == i + 1
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            pert[i] = x[mask].abs().sum()
        return pert.norm(p=0)


    def group_thres(self, x, x0norm, max_update):
        '''
        Sets the perturbation for all superpixels except for the k with the 
        largest 2-norm to zero.
        '''
        B, C, W, H = x.shape
        norms = torch.zeros((self.groups.max(),), device=x.device, dtype=torch.float)
        for i in range(self.groups.max()):
            mask = self.groups == i + 1
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            norms[i] = (x[mask] ** 2).sum()
        _, idx = norms.topk(k=x0norm.int() + max_update)
        res = torch.zeros_like(x)
        for i in idx:
            mask = (self.groups == i.item() + 1)
            mask = mask[None, :, :].repeat(1, C, 1, 1)
            res[mask] = x[mask]
        return res


    def nmAPG(self, x0, original_img, lambda1, search_lambda_inc,
              search_lambda_dec, target_class, max_update, oi=0):

        x0_norm0 = self.pert_groups(x0)

        temp = Variable(x0.data, requires_grad=False)
        logist = self.model(temp+original_img.data)
        if self.targeted:
            if self.loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss = ce(logist, torch.ones(1).long().to(self.device)*target_class)
            elif self.loss_type == 'cw':
                Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        else:
            Loss = self.CWLoss(logist, torch.ones(1).long().to(self.device)*target_class)
        self.model.zero_grad()

        z = x0
        y_pre = torch.zeros(original_img.shape).to(self.device)

        pre_loss = 0
        cur_loss = 0

        counter = 0
        success = 0

        alpha_y = 1e-3
        alpha_x = 1e-3

        alpha_min = 1e-20
        alpha_max = 1e20
        x_pre = x0
        x = x0
        t = 1
        t_pre = 0
        c = Loss + lambda1*torch.norm(x.data,0)
        q = 1
        k = 0
        while True:
            y = x + t_pre/t*(z-x) + (t_pre-1)/t*(x-x_pre)

            if k > 0:
                s = y - y_pre.data

                #gradient of yk
                temp_y = Variable(y.data, requires_grad=True)
                logist_y = self.model(temp_y+original_img.data)
                if self.targeted:
                    if self.loss_type == 'ce':
                        ce = torch.nn.CrossEntropyLoss()
                        Loss_y = ce(logist_y, torch.ones(1).long().to(self.device)*target_class)
                    elif self.loss_type == 'cw':
                        Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                else:
                    Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                self.model.zero_grad()
                if temp_y.grad is not None:
                    temp_y.grad.data.fill_(0)
                Loss_y.backward()
                grad_y = temp_y.grad

                #gradient of yk-1
                temp_y_pre = Variable(y_pre.data, requires_grad=True)
                logist_y_pre = self.model(temp_y_pre+original_img.data)
                if self.targeted:
                    if self.loss_type == 'ce':
                        ce = torch.nn.CrossEntropyLoss()
                        Loss_y_pre = ce(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                    elif self.loss_type == 'cw':
                        Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                else:
                    Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                self.model.zero_grad()
                if temp_y_pre.grad is not None:
                    temp_y_pre.grad.data.fill_(0)
                Loss_y_pre.backward()
                grad_y_pre = temp_y_pre.grad

                r = grad_y - grad_y_pre

                #prevent error caused by numerical inaccuracy
                if torch.norm(s,1) < 1e-5:
                    s = torch.ones(1).to(self.device)*1e-5

                if torch.norm(r,1) < 1e-10:
                    r = torch.ones(1).to(self.device)*1e-10

                alpha_y = torch.sum(s*r)/torch.sum(r*r)
                alpha_y = alpha_y.item()

            temp_alpha = alpha_y

            if temp_alpha < alpha_min:
                temp_alpha = alpha_min

            if temp_alpha > alpha_max:
                temp_alpha = alpha_max

            if np.isnan(temp_alpha):
                temp_alpha = alpha_min
            alpha_y = temp_alpha

            count1 = 0
            while True:
                count1 = count1 + 1
                if count1 > 1000:
                    break

                temp_y = Variable(y.data, requires_grad=True)
                logist_y = self.model(temp_y+original_img.data)
                if self.targeted:
                    if self.loss_type == 'ce':
                        ce = torch.nn.CrossEntropyLoss()
                        Loss_y = ce(logist_y,torch.ones(1).long().to(self.device)*target_class)
                    elif self.loss_type == 'cw':
                        Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                else:
                    Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
                self.model.zero_grad()
                if temp_y.grad is not None:
                    temp_y.grad.data.fill_(0)
                Loss_y.backward()
                grad_y = temp_y.grad

                z = self.prox_pixel(x=y-alpha_y*grad_y,alpha=alpha_y,
                                    lambda1=lambda1,original_img=original_img)

                #increase lambda
                if(search_lambda_inc == 1):
                    if(torch.norm(z,1) != 0):
                        return 0
                    else:
                        return 1

                #decrease lambda
                if(search_lambda_dec == 1):
                    if(torch.norm(z,1) == 0):
                        return 0
                    else:
                        return lambda1

                alpha_y = alpha_y * self.rho
                cond1 = self.F(z, lambda1, original_img,target_class) <= self.F(y, lambda1, original_img,target_class) - self.delta*(torch.norm(z-y,2)*torch.norm(z-y,2))
                cond2 = self.F(z, lambda1, original_img,target_class) <= c - self.delta*(torch.norm(z-y,2)*torch.norm(z-y,2))

                if(cond1 | cond2):
                    break
            if self.ver:
                print(f'\rHomotopy iteration {oi}, nmAPG iteration {k+1}, norm {y.norm(0)}', end='')
            if self.F(z, lambda1, original_img,target_class) <= c - self.delta*(torch.norm(z-y,2)*torch.norm(z-y,2)):
                x_pre = x
                temp_norm0 = self.pert_groups(z)
                if torch.abs(temp_norm0 - x0_norm0) > max_update:
                    z = self.group_thres(z, x0_norm0, max_update)
                    x = z
                else:
                    x = z
            else:

                if k > 0:
                    s = x - y_pre.data

                    temp_x = Variable(x.data, requires_grad=True)
                    logist_x = self.model(temp_x+original_img.data)
                    if self.targeted:
                        if self.loss_type == 'ce':
                            ce = torch.nn.CrossEntropyLoss()
                            Loss_x = ce(logist_x,torch.ones(1).long().to(self.device)*target_class)
                        elif self.loss_type == 'cw':
                            Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device)*target_class)
                    else:
                        Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device)*target_class)
                    self.model.zero_grad()
                    if temp_x.grad is not None:
                        temp_x.grad.data.fill_(0)
                    Loss_x.backward()
                    grad_x = temp_x.grad

                    temp_y_pre = Variable(y_pre.data, requires_grad=True)
                    logist_y_pre = self.model(temp_y_pre+original_img.data)
                    if self.targeted:
                        if self.loss_type == 'ce':
                            ce = torch.nn.CrossEntropyLoss()
                            Loss_y_pre = ce(logist_y_pre,torch.ones(1).long().to(self.device)*target_class)
                        elif self.loss_type == 'cw':
                            Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                    else:
                        Loss_y_pre = self.CWLoss(logist_y_pre, torch.ones(1).long().to(self.device)*target_class)
                    self.model.zero_grad()
                    if temp_y_pre.grad is not None:
                        temp_y_pre.grad.data.fill_(0)
                    Loss_y_pre.backward()
                    grad_y_pre = temp_y_pre.grad

                    r = grad_x - grad_y_pre

                    if torch.norm(s, 1) < 1e-5:
                        s = torch.ones(1).to(self.device) * 1e-5

                    if torch.norm(r,1) < 1e-10:
                        r = torch.ones(1).to(self.device)*1e-10

                    alpha_x = torch.sum(s*r)/torch.sum(r*r)
                    alpha_x = alpha_x.item()

                temp_alpha = alpha_x


                if temp_alpha < alpha_min:
                    temp_alpha = alpha_min

                if temp_alpha > alpha_max:
                    temp_alpha = alpha_max
                if np.isnan(temp_alpha):
                    temp_alpha = alpha_min
                alpha_x = temp_alpha

                count2 = 0
                while True:
                    count2 = count2 + 1

                    if count2 > 10:
                        break

                    temp_x = Variable(x.data, requires_grad=True)
                    logist_x = self.model(temp_x + original_img.data)
                    if self.targeted:
                        if self.loss_type == 'ce':
                            ce = torch.nn.CrossEntropyLoss()
                            Loss_x = ce(logist_x, torch.ones(1).long().to(self.device) * target_class)
                        elif self.loss_type == 'cw':
                            Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device) * target_class)
                    else:
                        Loss_x = self.CWLoss(logist_x, torch.ones(1).long().to(self.device) * target_class)
                    self.model.zero_grad()
                    if temp_x.grad is not None:
                        temp_x.grad.data.fill_(0)
                    Loss_x.backward()
                    grad_x = temp_x.grad

                    v = self.prox_pixel(x=x-alpha_x*grad_x,alpha=alpha_x,lambda1=lambda1,original_img=original_img)
                    alpha_x = self.rho * alpha_x
                    cond3 = self.F(v, lambda1, original_img,target_class) <= c - self.delta*(torch.norm(v-x,2)*torch.norm(v-x,2))

                    if cond3:
                        break
                    if torch.abs(self.F(v, lambda1, original_img,target_class) - (c - self.delta*(torch.norm(v-x,2)*torch.norm(v-x,2)))) < 1e-3:
                        break


                if self.F(z, lambda1, original_img,target_class) <= self.F(v, lambda1, original_img,target_class):
                    x_pre = x
                    temp_norm0 = self.pert_groups(z)
                    if torch.abs(temp_norm0 - x0_norm0) > max_update:
                        z = self.group_thres(z, x0_norm0, max_update)
                        x = z
                    else:
                        x = z
                else:
                    x_pre = x
                    temp_norm0 = self.pert_groups(v)
                    if torch.abs(temp_norm0 - x0_norm0) > max_update:
                        z = self.group_thres(v, x0_norm0, max_update)
                        x = v
                    else:
                        x = v


            thres = self.max_epsilon
            x = torch.clamp(x.data,-thres,thres)
            temp_img = original_img.data + x.data
            temp_img = torch.clamp(temp_img.data, *self.img_range)
            x = temp_img.data - original_img.data

            y_pre = y.data
            t = (np.sqrt(4*t*t+1)+1)/2
            q = self.eta*q + 1
            c = (self.eta*q*c + self.F(x, lambda1, original_img, target_class))/q

            logist = self.model(x.data+original_img.data)
            _,target=torch.max(logist,1)

            k = k + 1

            pre_loss = cur_loss

            if not self.targeted:
                cur_loss = self.CWLoss(logist.data, torch.ones(1).long().to(self.device)*target_class).item()
            else:
                if self.loss_type == 'cw':
                    cur_loss = self.CWLoss(logist.data, torch.ones(1).long().to(self.device)*target_class).item()
                else:
                    ce = torch.nn.CrossEntropyLoss()
                    cur_loss = ce(logist.data, torch.ones(1).long().to(self.device) * target_class).item()
            self.model.zero_grad()

            #success
            if self.targeted:
                if(target == target_class):
                    success = 1
                    break
            else:
                if((target != target_class)):
                    success = 1
                    break

            if ((success == 0) and (k >= self.maxiter) and (np.abs(pre_loss-cur_loss) < 1e-3) and (counter==1)):
                break

            if((k >= self.maxiter) and (np.abs(pre_loss-cur_loss) < 1e-3)):
                counter = 1

        return x, success


    def lambda_test(self, grad, lam, original_img):
        '''
        Check if initial perturbation is all zero for given lambda.
        '''
        znorm = torch.norm(self.prox_pixel(-1e-3 * grad, 1e-3, lam, original_img), p=1, dim=(1,2,3))
        return 0 if znorm != 0 else 1


    def search_lambda(self, original_img, target_class):
        '''
        Initialize lambda.
        '''

        temp_y = Variable(torch.zeros_like(original_img), requires_grad=True)
        logist_y = self.model(temp_y+original_img.data)
        if self.targeted:
            if self.loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss_y = ce(logist_y, torch.ones(1).long().to(self.device)*target_class)
            elif self.loss_type == 'cw':
                Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
        else:
            Loss_y = self.CWLoss(logist_y, torch.ones(1).long().to(self.device)*target_class)
        self.model.zero_grad()
        if temp_y.grad is not None:
            temp_y.grad.data.fill_(0)
        Loss_y.backward()
        grad = temp_y.grad

        lam = self.beta
        while True:
            if not self.lambda_test(grad, lam, original_img):
                lam += self.beta
            else:
                break
        while True:
            if self.lambda_test(grad, lam, original_img):
                lam *= 0.99
            else:
                break
        
        if self.ver:
            print(f'Lambda = {lam * self.val_c}')
        return lam * self.val_c

    def get_grid_segmentation(self,a,n):
            """
            Creates an (H, W) tensor where pixels are grouped into an n x n grid.
            
            Args:
                H, W: Height and Width of the image.
                n: Number of blocks along each dimension (total groups = n*n).
            """

            H ,W= a.shape[2:]
            block_h = H // n
            block_w = W// n
            
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H), 
                torch.arange(W), 
                indexing='ij'
            )
            
            row_idx = torch.div(y_coords, block_h, rounding_mode='floor').clamp(max=n)
            col_idx = torch.div(x_coords, block_w, rounding_mode='floor').clamp(max=n)
            
            grid_indices = row_idx * n + col_idx
            
            return grid_indices.long()+1


    def homotopy(self, original_img, target_class):

        # self.groups = torch.from_numpy(skimage.segmentation.slic(original_img.cpu().numpy(),
        #                                                          n_segments=self.n_segments,
        #                                                          channel_axis=1)).to(self.device)
        #print(self.get_grid_segmentation(original_img,n=15).shape)
        self.groups = self.get_grid_segmentation(original_img,n=15).to(self.device)

        lambda1 = self.search_lambda(original_img, target_class)

        x = torch.zeros(original_img.shape).to(self.device)
        pre_norm0 = 0
        cur_norm0 = 0

        max_norm0 = torch.norm(torch.ones(x.shape).to(self.device),0).item()
        outer_iter = 0
        max_update = self.max_update

        while True:
            print(outer_iter)
            outer_iter = outer_iter + 1
            x, success = self.nmAPG(x0=x, original_img=original_img, lambda1=lambda1, search_lambda_inc=0,
                               search_lambda_dec=0, target_class=target_class,
                               max_update=max_update, oi=outer_iter)
            max_update = self.max_update
            pre_norm0 = cur_norm0
            cur_norm0 = torch.norm(torch.ones(x.shape).to(self.device)*x.data,0).item()
            cur_norm1 = torch.norm(torch.ones(x.shape).to(self.device) * x.data, 1).item()

            #attack fail
            if(cur_norm0 > max_norm0*0.95 and outer_iter*max_update > max_norm0*0.95):
                break

            iters = 0
            if (cur_norm1 <= cur_norm0 * self.max_epsilon * self.val_gamma):
                max_update = 1
                p = bisect.bisect_left(self.iter_inc, cur_norm0)
                iters = (p + 1) * self.iter_init

            if success == 0:
                x = self.after_attack(x, original_img, target_class, post=0, iters=iters)
                lambda1 = self.dec_factor * lambda1
            else:
                break

            logi = self.model(x.data+original_img.data)
            _,cur_class=torch.max(logi,1)
            if self.targeted:
                if((cur_class == target_class)):
                    break
            else:
                if((cur_class != target_class)):
                    break

        x = self.after_attack(x, original_img, target_class, post=1, iters=iters)
        #print('')
        return x


    def CWLoss(self, logits, target):
        target = torch.ones(logits.size(0)).type(torch.float).to(self.device).mul(target.float())
        target_one_hot = Variable(torch.eye(logits.size(1)).type(torch.float).to(self.device)[target.long()])

        real = torch.sum(target_one_hot*logits, 1)
        other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        kappa = torch.zeros_like(other).fill_(self.kappa)

        if self.targeted:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))



################################### SAPF ######################################

class SAPF(Attack):
    def __init__(self, model, *args, targeted=True, img_range=(-1, 1), iters=10, 
                 iters_delta=200, iters_G=200, etas=(0.1, 0.1), eta_decay=0.9,
                 rhos=(5e-3, 5e-3, 5e-3, 1e-4), rhos_max=(20, 20, 100, 0.01),
                 k=350, lam1=1e-3, lam2=10, lam1_bounds=(0, 100), loss='cw',
                 nsegs=150, search_steps=10, sequential=False, **kwargs):
        """
        Implementation of Sparse Adversarial Attack via Perturbation Factorization 
        https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670035.pdf
        Authors' implementation: https://github.com/wubaoyuan/Sparse-Adversarial-Attack/

        args:
        model:         Callable, PyTorch classifier.
        img_range:     Tuple of int/float, lower and upper bound of image entries.
        targeted:      Bool, given label is used as a target label if True.
        iters:         Int, number of iterations.
        iters_delta:   Int, number of iterations for the delta update.
        iters_G:       Int, number of iterations for the G update.
        etas:          Tuple of float, step sizes for delta / G update.
        eta_decay:     Float, decay factor for the step sizes.
        rhos:          Tuple of float, ADMM penalty parameters.
        rhos_max:      Tuple of float, maximum values for the ADMM parameters.
        k:             Int, sparsity parameter.
        lam1:          Float, inverse of l2 trade-off parameter.
        lam2:          Float, group-wise sparsity trade-off parameter.
        lam1_bounds:   Tuple of float, lower and upper bound for binary sarch in lam1.
        loss:          String, 'ce': cross entropy loss, 'cw': Carlini-Wagner loss.
        nsegs:         Int, number of segments for SLIC.
        search_steps:  Int, number of binary search steps wrt. lam1.
        sequential:    Bool, perturbations are computed sequentially for all
                       images in the batch if True. For fair comparison to
                       Homotopy attack.
        """

        super().__init__(model, targeted, img_range)
        self.iters = iters
        self.iters_delta = iters_delta
        self.iters_G = iters_G
        self.etas = etas
        self.eta_decay = eta_decay
        self.rhos = rhos
        self.rhos_max = rhos_max
        self.k = k
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam1_bounds = lam1_bounds
        self.nsegs = nsegs
        self.search_steps = search_steps
        self.sequential = sequential

        if loss == 'ce':
            self.lossfn = lambda x, y: (2 * self.targeted - 1) * F.cross_entropy(self.model(x), y)
        elif loss == 'cw':
            self.lossfn = self.__CWLoss
        else:
            raise NotImplementedError(f'No loss function implemented for loss = {loss}')



    def __call__(self, x, y, *args, **kwargs):
        '''
        Call the attack for a batch of images x or sequentially for all images
        in x depending on self.sequential.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        if self.sequential:
            result = x.clone()
            for i, (x_, y_) in enumerate(zip(x, y)):
                result[i] = self.__perform_att(x_.unsqueeze(0),
                                               y_.unsqueeze(0)).detach()
            return result
        else:
            return self.__perform_att(x, y)

    def get_grid_segmentation(self,a,n):
        """
        Creates an (H, W) tensor where pixels are grouped into an n x n grid.
        
        Args:
            H, W: Height and Width of the image.
            n: Number of blocks along each dimension (total groups = n*n).
        """

        H ,W= a.shape[2:]
        block_h = H // n
        block_w = W// n
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H), 
            torch.arange(W), 
            indexing='ij'
        )
        
        row_idx = torch.div(y_coords, block_h, rounding_mode='floor').clamp(max=n)
        col_idx = torch.div(x_coords, block_w, rounding_mode='floor').clamp(max=n)
        
        grid_indices = row_idx * n + col_idx
        
        return grid_indices.long()+1
            


    def __perform_att(self, x, y):
        '''
        Perform the attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples.
        '''

        # save a list with one tensor per image. each tensor is a batch of [C, H, W]
        # masks, one for each segment/super pixel in the image
        segments = []
        for idx in range(x.size(0)):
            nprgb = np.array(x[idx].permute(1, 2, 0).cpu(), dtype=np.uint8)
            segs = torch.from_numpy(skimage.segmentation.slic(nprgb, n_segments=self.nsegs, compactness=10))
            segmask = torch.zeros((segs.max()-segs.min()+1, *x.shape[1:]), dtype=x.dtype)
            for i in range(segmask.size(0)):
                segmask[i, :, segs==i+1] = 1
            segments.append(segmask.to(self.device))

        x = x.to(self.device)
        y = y.to(self.device)
        result = x.clone()
        # tensors for binary search
        best_l0 = torch.full_like(y, torch.inf, dtype=torch.float)
        active = torch.ones_like(y, dtype=torch.bool)
        lam1 = torch.full_like(y, self.lam1, dtype=torch.float)
        lam_lb = torch.full_like(y, self.lam1_bounds[0], dtype=torch.float)
        lam_ub = torch.full_like(y, self.lam1_bounds[1], dtype=torch.float)

        for i in range(self.search_steps):
            if x.size(0) == 0:
                break
            print(i+1)
            delta, G = self.__perturb(x, y, lam1.view(-1, 1, 1, 1), segments)
            
            # clamp adv. example to valid range
            x_adv = torch.clamp(x + delta * G, *self.img_range)
            preds = torch.argmax(self.model(x_adv), dim=1)
            mask = preds == y if self.targeted else preds != y

            # binary search on lam1
            for j in range(x.size(0)):
                if mask[j]:
                    lam_ub[j] = min(lam1[j], lam_ub[j])
                    if lam_ub[j] < self.lam1_bounds[1]:
                        lam1[j] = (lam_ub[j] + lam_lb[j]) / 2
                    l0 = torch.norm(x_adv[j] - x[j], p=0)
                    if l0 < best_l0[j]:
                        result[active.nonzero()[j]] = x_adv[j]
                        best_l0[j] = l0
                else:
                    lam_lb[j] = max(lam_lb[j], lam1[j])
                    if lam_ub[j] < self.lam1_bounds[1]:
                        lam1[j] = (lam_ub[j] + lam_lb[j]) / 2
                    else:
                        lam1[j] *= 10

            mask = lam1 < 0.01 * self.lam1
            # stop search if lam1 falls below minimum value
            if mask.any():
                mask = torch.logical_not(mask)
                active[active.clone()] = mask
                x, y = x[mask], y[mask]
                lam1, lam_lb, lam_ub = lam1[mask], lam_lb[mask], lam_ub[mask]
                segments = [segments[j] for j in range(len(segments)) if j in mask.nonzero()]
                best_l0 = best_l0[mask]

        return result.detach()


    def __perturb(self, x, y, lam1, segments):
        '''
        Compute adversrial perturbation for a given trade-off parameter lam1.
        '''
        G = torch.ones_like(x)
        delta = torch.zeros_like(x)
        eta_delta, eta_G = self.etas
        rhos = self.rhos

        for i in tqdm(range(self.iters)):
            delta, eta_delta = self.__update_delta(x, y, delta, G, eta_delta, lam1, False)
            G, rhos, eta_G = self.__update_G(x, y, delta, G, rhos, eta_G, lam1, segments, i)
        G = (G > 0.5).float()
        delta, _ = self.__update_delta(x, y, delta, G, eta_delta, lam1, True)

        return delta, G


    def __update_delta(self, x, y, delta, G, eta, lam1, finetune):
        '''
        Update delta via gradient descent.
        '''
        iters = int(self.iters_delta / 2) if finetune else self.iters_delta

        for j in range(iters):
            delta.requires_grad = True
            loss = self.lossfn(torch.clamp(x + delta * G, *self.img_range), y)
            loss.backward()
            grad = delta.grad.data

            with torch.no_grad():
                grad = 2 * delta * G * G + lam1 * grad
                delta = delta - eta * grad

                if not j % 50:
                    eta = max(eta * self.eta_decay, 0.001)

        return delta, eta


    def __update_G(self, x, y, delta, G, rhos, eta, lam1, segments, it):
        '''
        Update G via ADMM.
        '''
        rho1, rho2, rho3, rho4 = rhos
        # y1, z1: box constraint, y2, z2: l2 sphere, y3, z3: group sparsity, z4: sparsity
        y1, y2, y3 = torch.ones_like(G), torch.ones_like(G), torch.ones_like(G)
        z1, z2, z3 = torch.zeros_like(G), torch.zeros_like(G), torch.zeros_like(G)
        z4 = torch.zeros((G.size(0), 1, 1, 1), device=self.device, dtype=G.dtype)

        for j in range(self.iters_G):
            # update primals
            with torch.no_grad():
                y1 = torch.clamp(G + z1 / rho1, 0, 1)
                y2 = self.__project_l2(G + z2 / rho2)
                for idx in range(G.size(0)):
                    seg = (G[idx] + z3[idx] / rho3) * segments[idx]
                    norm = torch.norm(seg, p=2, dim=(1,2,3), keepdim=True)
                    coeff = torch.clamp(1 - self.lam2 / (rho3 * norm), 0)
                    y3[idx] = torch.sum(coeff * seg, dim=0)

            # update G
            G.requires_grad = True
            loss = self.lossfn(torch.clamp(x + delta * G, *self.img_range), y)
            loss.backward()
            grad = G.grad.data

            with torch.no_grad():
                grad = 2 * G * delta * delta + lam1 * grad
                grad = grad + z1 + z2 + z3 + z4
                grad = grad + rho1 * (G - y1) + rho2 * (G - y2) + rho3 * (G - y3)
                grad = grad + rho4 * (torch.sum(G, dim=(1,2,3), keepdim=True) - self.k)
                G = G - eta * grad

                # update duals
                z1 += rho1 * (G - y1)
                z2 += rho2 * (G - y2)
                z3 += rho3 * (G - y3)
                z4 += rho4 * (torch.sum(G, dim=(1,2,3), keepdim=True) - self.k)

                rho1 = min(1.01 * rho1, self.rhos_max[0])
                rho2 = min(1.01 * rho2, self.rhos_max[1])
                rho3 = min(1.01 * rho3, self.rhos_max[2])
                rho4 = min(1.01 * rho4, self.rhos_max[3])

                if not j % 50:
                    eta = max(eta * self.eta_decay, 0.001)

        return G, (rho1, rho2, rho3, rho4), eta


    def __project_l2(self, x):
        '''
        Project x on an l2 sphere translated by the all 0.5 vector.
        '''
        trans = torch.full_like(x, 0.5)
        norm = torch.norm(x - trans, p=2, dim=(1, 2, 3), keepdim=True)
        return (math.sqrt(float(x[0].numel())) / 2) * ((x - trans) / norm) + trans
                

    def __CWLoss(self, x, y, kappa=0):
        '''
        CW loss function.
        '''
        logits = self.model(x)
        one_hot_y = F.one_hot(y, logits.size(1))
        Z_t = torch.sum(logits * one_hot_y, dim=1)
        Z_i = torch.amax(logits * (1 - one_hot_y) - (one_hot_y * 1e5), dim=1)
        return F.relu((2 * self.targeted - 1) * (Z_i - Z_t) + kappa).sum()
    


################################## SparseRS #####################################

class SparseRS(Attack):
    def __init__(self, model, *args, targeted=False, img_range=(-1, 1),
                 n_queries=10000, k=100, n_restarts=10, alpha_init=0.8, **kwargs):
        '''
        Implementation of the L0 variant SparseRS https://arxiv.org/abs/2006.12834
        Authors' implementation: https://github.com/fra31/sparse-rs
        
        args:
        model:         Callable, PyTorch classifier.
        targeted:      Bool, given label is used as a target label if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        n_queries:     Int, max number of queries to the model
        k:             Int, initial sparsity parameter
        n_restarts:    Int, number of restarts with random initialization
        alpha_init:    Float, inital value for alpha schedule
        '''
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.n_queries = n_queries
        self.k = k
        self.n_restarts = n_restarts
        self.alpha_init = alpha_init


    def __call__(self, x, y, *args, **kwargs):
        '''
        Perform SparseRS L0 on a batch of images x with corresponding labels y.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''
        torch.random.manual_seed(0)
        torch.cuda.random.manual_seed(0)
        x = x.to(self.device)
        y = y.to(self.device)
        active = torch.ones_like(y, dtype=torch.bool)
        result = x.clone()

        with torch.no_grad():
            for _ in range(self.n_restarts):
                if len(x) == 0:
                    break

                x_adv = self.__perturb(x.clone(), y.clone())
                preds = torch.argmax(self.model(x_adv), dim=1)
                mask = preds == y if self.targeted else preds != y
                # save successful examples
                if mask.any():
                    tmp = result[active]
                    tmp[mask] = x_adv[mask]
                    result[active] = tmp
                    mask = torch.logical_not(mask)
                    active[active.clone()] = mask
                    x, y = x[mask], y[mask]

        return result.detach()
    

    def __perturb(self, x, y):
        '''
        Perform the attack from a random starting point.
        '''
        B, C, H, W = x.shape
        batchidx = torch.arange(B, device=self.device).view(-1, 1)
        result = x.clone()
        active = torch.ones_like(y, dtype=torch.bool)

        # M: set of perturbed pixel indices, U_M: set of unperturbed pixel indices
        batch_randperm = torch.rand(B, H * W, device=self.device).argsort(dim=1)
        M = batch_randperm[:, :self.k]
        U_M = batch_randperm[:, self.k:]
        result[batchidx, :, M//W, M%H] = self.__sampleDelta(B, C, self.k)

        best_margin, best_loss = self.__lossfn(result, y)

        for i in range(1, self.n_queries):
            if B == 0:
                break
            # reset k_i currently perturbed pixels and perturb k_i new pixels
            k_i = max(int(self.__alphaSchedule(i) * self.k), 1)
            A_idx = torch.randperm(self.k, device=self.device)[:k_i]
            B_idx = torch.randperm(H * W - self.k, device=self.device)[:k_i]
            A_set, B_set = M[:, A_idx], U_M[:, B_idx]

            z = result[active].clone()
            z[batchidx, :, A_set//W, A_set%H] = x[batchidx, :, A_set//W, A_set%H]
            if k_i > 1:
                z[batchidx, :, B_set//W, B_set%H] = self.__sampleDelta(B, C, k_i)
            else: # if only one pixel is changed, make sure it actually changes
                new_color = self.__sampleDelta(B, C, k_i)
                while (mask := (z[batchidx, :, B_set//W, B_set%H] == new_color).view(B, -1).all(dim=-1)).any():
                    new_color[mask] = self.__sampleDelta(mask.int().sum().item(), C, k_i)
                z[batchidx, :, B_set//W, B_set%H] = new_color

            # save perturbations that improved the loss/margin
            margin, loss = self.__lossfn(z, y)
            mask = loss < best_loss
            best_loss[mask] = loss[mask]
            mask = torch.logical_or(mask, margin < -1e-6)
            if mask.any():
                best_margin[mask] = margin[mask]
                tmp = result[active]
                tmp[mask] = z[mask]
                result[active] = tmp
                U_M[mask.nonzero().view(-1, 1), B_idx] = A_set[mask]
                M[mask.nonzero().view(-1, 1), A_idx] = B_set[mask]
            
            # stop working on successful adv examples
            mask = best_margin < 0
            if mask.any():
                mask = torch.logical_not(mask)
                active[active.clone()] = mask
                x, y, z, M, U_M = x[mask], y[mask], z[mask], M[mask], U_M[mask]
                best_margin, best_loss = best_margin[mask], best_loss[mask]
                B = len(y)
                batchidx = torch.arange(B, device=self.device).view(-1, 1)

        return result


    def __sampleDelta(self, B, C, k):
        '''
        Sample k-pixel perturbations for B images. Each pixel is assigned a
        random corner in the C-dimensional cube defined by self.img_range.
        '''
        fac = self.img_range[1] - self.img_range[0]
        return self.img_range[0] + fac * torch.randint(0, 1, [B, k, C],
                                                       dtype=torch.float,
                                                       device=self.device)
    

    def __alphaSchedule(self, iteration):
        '''
        Update number of pixels to perturb based in the current iteration.
        '''
        iteration = int(iteration / self.n_queries * 10000)
        factors = [1, 2, 4, 5, 6, 8, 10, 12, 15, 20]
        alpha_schedule = [10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000]
        idx = bisect.bisect_left(alpha_schedule, iteration)
        return self.alpha_init / factors[idx]
    

    def __lossfn(self, x, y):
        '''
        Compute the loss depending on self.targeted.
        '''
        out = self.model(x)
        ce = F.cross_entropy(out, y, reduction='none')
        corr = out[range(len(out)), y]
        out[range(len(out)), y] = -torch.inf
        other = out.max(dim=-1)[0]
        tfac = 2 * self.targeted - 1
        return tfac * (other - corr), tfac * ce
    


################################### PGD00 ######################################

class PGD0(Attack):
    def __init__(self, model, *args, img_range=(-1, 1), k=100, n_restarts=5,
                 targeted=False, iters=200, stepsize=120000/255., **kwargs):
        '''
        Implementation of the PGD0 attack https://arxiv.org/pdf/1909.05040
        Authpr's implementation: https://github.com/fra31/sparse-imperceivable-attacks/tree/master

        args:
        model:         Callable, PyTorch classifier.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        k:             Int, sparsity parameter.
        n_restarts:    Int, number of restarts from random perturbation.
        iters:         Int, number of gradient descent steps per restart.
        stepsize:      Float, step size for gradient descent.
        '''
        super().__init__(model, img_range=img_range, targeted=targeted)
        self.k = k
        self.n_restarts = n_restarts
        self.iters = iters
        self.stepsize = stepsize


    def __call__(self, x, y, *args, **kwargs):
        '''
        Perform the PGD_0 attack on a batch of images x.

        args:
        x:   Tensor of shape [B, C, H, W], batch of images.
        y:   Tensor of shape [B], batch of labels.

        Returns a tensor of the same shape as x containing adversarial examples
        '''

        x = x.to(self.device)
        y = y.to(self.device)
        result = x.clone()
        active = torch.ones_like(y, dtype=torch.bool)

        for _ in range(self.n_restarts):
            if not len(x):
                break
            lb, ub = self.img_range[0] - x, self.img_range[1] - x
            pert = torch.clamp(x + (ub - lb) * torch.rand_like(x) + lb, *self.img_range) - x
            pert = self.project_L0(pert, lb, ub)

            for _ in range(self.iters):
                pert.requires_grad = True
                loss = self.lossfn(x + pert, y)
                loss.backward()
                grad = pert.grad.data
                with torch.no_grad():
                    grad /= grad.abs().sum(dim=(1,2,3), keepdim=True) + 1e-10
                    pert += (torch.rand_like(x) - .5) * 1e-12 - self.stepsize * grad
                    pert = self.project_L0(pert, lb, ub)

            pred = torch.argmax(self.model(x + pert), dim=1)
            mask = pred == y if self.targeted else pred != y
            if mask.any():
                tmp = result[active]
                tmp[mask] = x[mask] + pert[mask]
                result[active] = tmp
                mask = torch.logical_not(mask)
                active[active.clone()] = mask
                x, y = x[mask], y[mask]

        return result
    

    def project_L0(self, pert, lb, ub):
        '''
        Project a batch of perturbations such that at most self.k pixels
        are perturbed and componentwise there holds lb <= pert <= ub.
        '''
        B, C, H, W = pert.shape
        p1 = torch.sum(pert ** 2, dim=1)
        p2 = torch.clamp(torch.minimum(ub - pert, pert - lb), 0)
        p2 = torch.sum(p2 ** 2, dim=1)
        p3 = torch.topk(-1 * (p1 - p2).view(p1.size(0), -1), k=H*W-self.k, dim=-1)[1]
        pert = torch.maximum(torch.minimum(pert, ub), lb)
        pert[torch.arange(0, B).view(-1, 1), :, p3//W, p3%H] = 0
        return pert


    def lossfn(self, x, y):
        '''
        Compute the loss at x, y.
        '''
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        return (2 * self.targeted - 1) * loss
