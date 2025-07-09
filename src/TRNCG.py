import torch
import torch.optim as optim
import torch.autograd as autograd

from typing import NewType, List, Tuple


Tensor = NewType('Tensor', torch.Tensor)


class TrustRegion(optim.Optimizer):

    def __init__(self, params: List[Tensor], delta_max: float = 10, delta0: float = .005,
        eta: float = 0.125, gtol: float = 1e-10, epsilon: float = 1e-9, max_iter_cg: int = 1000) -> None:

        defaults = dict()

        super(TrustRegion, self).__init__(params, defaults)

        self.steps = 0
        self.delta_max = delta_max
        self.delta0 = delta0
        self.eta = eta
        self.gtol = gtol
        self._params = self.param_groups[0]['params']
        self.epsilon = epsilon
        self.max_iter_cg = max_iter_cg

    @torch.enable_grad()
    def _compute_hessian_vector_product(self, gradient: Tensor, p: Tensor) -> Tensor:
        inner_prod = torch.dot(gradient, p)
        hvp = autograd.grad(inner_prod, self._params, only_inputs=True, retain_graph=True, allow_unused=True)
        return torch.cat([torch.flatten(vp) for vp in hvp], dim=-1)


    def _gather_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        output = torch.cat(views, 0)
        return output

    @torch.no_grad()
    def _compute_rho(self, p, start_loss, g, closure):
        hvp = self._compute_hessian_vector_product(g, p)
        # Use a torch.no_grad() context since we are updating the parameters in place
        with torch.no_grad():
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = p[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param))
                start_idx += num_els

        # only need the value of the loss at the new point to find the ratio of the actual and the expected improvement
        new_loss= closure(backward=False)
        #  actual loss decrease
        numerator = start_loss - new_loss
        new_quad_val = self._quad_model(p, start_loss, g, hvp)

        denominator = start_loss - new_quad_val

        rho = numerator / denominator
        return rho

    @torch.no_grad()
    def _quad_model(self, p: Tensor, loss: float, g: Tensor, hvp: Tensor) -> float:
        term1 = torch.dot(torch.flatten(g), torch.flatten(p))
        term2 = torch.dot(torch.flatten(hvp), torch.flatten(p))
        quad_approx = loss + term1 + 0.5 * term2
        return quad_approx

    @torch.no_grad()
    def compute_tau(self, z: Tensor, d: Tensor, delta: float) -> Tuple[Tensor, Tensor]:
        a = torch.norm(d) ** 2
        b = 2 * torch.dot(d, z)
        c = torch.norm(z) ** 2 - delta ** 2
        sq_root = torch.sqrt(b * b - 4 * a * c)
        tau = (-b + sq_root) / (2 * a)
        return tau

    @torch.no_grad()
    def _CG_Steihaug(self, loss: float, flat_grad: Tensor, delta: float) -> Tuple[Tensor, bool]:
        max_iter_cg = self.max_iter_cg
        norm_g = torch.norm(flat_grad).item()
        eta = min(0.5, norm_g)
        e_k = eta * norm_g

        z = torch.zeros_like(flat_grad, requires_grad=False)
        r = flat_grad.detach()
        d = -r

        # If the norm of the gradients is smaller than the tolerance then exit
        if norm_g <= e_k:
            return z, False, 0, 1

        j = 0
        # Iterate to solve the subproblem
        while j <= max_iter_cg:
            j += 1
            # Calculate the Hessian-Vector product
            B_d = self._compute_hessian_vector_product(flat_grad, d)
            d_B_d = torch.dot(B_d, d)

            # If negative curvature
            if d_B_d.item() <= 0:
                tau = self.compute_tau(z, d, delta)
                p = z + tau * d
                return p, True, j, 2

            # get alpha
            normsqr_r = torch.norm(r) ** 2
            alpha = normsqr_r / d_B_d

            # Update the point
            z_new = z + alpha * d
            norm_z_new = torch.norm(z_new)

            # If the point is outside of the trust region project it on the border and return
            if norm_z_new.item() >= delta:
                tau = self.compute_tau(z, d, delta)
                p = z + tau * d
                return p, True, j, 3

            r_new = r + alpha * B_d

            # If the residual is small enough, exit
            if torch.norm(r_new).item() < e_k:
                return z_new, False, j, 1

            beta = (torch.norm(r_new) ** 2) / normsqr_r

            # new search direction
            d = (-r_new + beta * d).squeeze()
            z = z_new
            r = r_new
        return p, False, max_iter_cg, 4
        

    def step(self, closure=None, get_stop_cond=False) -> float:
        starting_loss = closure(backward=True)

        flat_grad = self._gather_flat_grad()

        state = self.state
        if len(state) == 0:
            state['delta'] = torch.full([1], self.delta0, dtype=flat_grad.dtype, device=flat_grad.device)
        delta = state['delta']

        p, hit_boundary, converge_iters, stopping_cond = self._CG_Steihaug(starting_loss, flat_grad, delta)
        self.p = p
        # print(converge_iters, 'convergence iterations, with stopping condition', stopping_cond)
        # check if gradient is zero
        if torch.norm(p).item() <= self.gtol:
            if get_stop_cond is True:
                return starting_loss, stopping_cond
            return starting_loss

        rho = self._compute_rho(p, starting_loss, flat_grad, closure)

        # sizing of the trust region
        if rho.item() < 0.25:
            delta = 0.25 * delta

        elif rho.item() > 0.75 and hit_boundary:
            delta = min(self.delta_max, 2 * delta)

        # accept or reject the step
        if rho.item() <= self.eta:
            # if we reject step, then undo the changes
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = p[start_idx:start_idx + num_els]
                param.data.add_(-curr_upd.view_as(param))
                start_idx += num_els

        self.steps += 1
        if get_stop_cond is True:
            return starting_loss, stopping_cond
        return starting_loss
