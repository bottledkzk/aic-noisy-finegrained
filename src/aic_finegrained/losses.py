"""Soft-target losses and the train-wide entropic transport dual."""

import copy

import torch
import torch.nn.functional as F

from .model import precision


def objective(logits,q,ls,coefficient):
    q=q.detach()
    logp=F.log_softmax(logits.float(),dim=1)
    p=logp.exp()
    target=(1-ls)*q+ls/q.shape[1]
    ce=-(target*logp).sum(1)
    erce=1-(p*q).sum(1)
    return (ce+coefficient*erce).mean(),ce,erce,p


def analytic_gradients(p,q,ls):
    q=q.detach();p=p.detach()
    target=(1-ls)*q+ls/q.shape[1]
    return p*target.sum(1,keepdim=True)-target,p*((p*q).sum(1,keepdim=True)-q)


@torch.no_grad()
def global_dual(log_kernel, prior, tol=2e-06, max_iter=100):
    """Solve the entropic-OT column dual, using damped Newton for sharp kernels.

 phi(v)=mean(logsumexp(log_kernel+v))-prior.dot(v).
 Gradient=column_mass-prior; Hessian=diag(column_mass)-mean(q q^T).
 Fixing the last dual to zero removes the harmless additive gauge.
 """
    p = log_kernel.double()
    pi = prior.double()
    pi = pi / pi.sum()
    n = len(p)
    assert p.ndim == 2 and pi.shape == (p.shape[1],) and torch.isfinite(p).all() and bool((pi > 0).all())
    v = torch.zeros_like(pi)
    for it in range(max_iter):
        q = torch.softmax(p + v, dim=1)
        mass = q.mean(0)
        err = float((mass / pi - 1).abs().max())
        if err < tol:
            break
        grad = mass - pi
        h = torch.diag(mass[:-1]) - q[:, :-1].T @ q[:, :-1] / n
        h += torch.eye(len(pi) - 1, device=h.device, dtype=h.dtype) * 1e-12
        direction = torch.cat([torch.linalg.solve(h, grad[:-1]), v.new_zeros(1)])
        descent = (grad * direction).sum()
        assert float(descent) > 0
        old = torch.logsumexp(p + v, dim=1).mean() - (pi * v).sum()
        step = 1.0
        for _ in range(40):
            proposed = v - step * direction
            value = torch.logsumexp(p + proposed, dim=1).mean() - (pi * proposed).sum()
            if float(value) <= float(old - 0.0001 * step * descent) + 1e-13:
                break
            step *= 0.5
        else:
            raise AssertionError('OT dual line search failed')
        v = proposed - proposed[-1]
    else:
        raise AssertionError('Full-train Sinkhorn did not converge')
    return (v.float(), {'iterations': it + 1, 'class_relative_error': err, 'row_error': float((q.sum(1) - 1).abs().max()), 'scope_rows': n, 'scope_classes': len(pi), 'solver': 'damped_Newton_same_entropic_OT_objective'})


class OnlineTeacher:
    """Maintain EMA logits for every training row and refresh the class dual each epoch."""

    def __init__(self, student, anchors):
        self.student = student
        self.teacher = copy.deepcopy(student).eval().requires_grad_(False)
        self.parameter_pairs = [(teacher, p) for teacher, p in
                                zip(self.teacher.parameters(), student.parameters(), strict=True)
                                if p.requires_grad]
        self.anchors = anchors
        self.memory = anchors.clamp_min(1e-8).log()
        self.prior = anchors.double().mean(0)
        self.dual, _ = global_dual(self.memory, self.prior)
        self.seen = torch.zeros(len(anchors), dtype=torch.bool, device=anchors.device)
        self.updates = 0

    @torch.no_grad()
    def target(self, images, ids, epoch):
        with precision(images.device):
            logits = self.teacher(images)
        logp = F.log_softmax(logits.float() / 0.5, dim=1)
        self.memory[ids] = logp
        self.seen[ids] = True
        beta = 0.7 * min(1.0, (epoch - 1) / 5.0)
        return (1 - beta) * self.anchors[ids] + beta * torch.softmax(logp + self.dual, dim=1)

    @torch.no_grad()
    def update(self):
        self.updates += 1
        decay = min(0.999, (1 + self.updates) / (10 + self.updates))
        for teacher, student in self.parameter_pairs:
            teacher.mul_(decay).add_(student.detach(), alpha=1 - decay)

    def finish_epoch(self):
        if not self.seen.all():
            raise RuntimeError("The EMA memory must see every training row within each epoch")
        self.dual, details = global_dual(self.memory, self.prior)
        self.seen.zero_()
        return details


class RobustSchedule:
    """Calibrate one class-balanced gradient scale for each independent SCE2 branch."""

    def __init__(self, classes, device):
        self.sums = torch.zeros(2, classes, dtype=torch.float64, device=device)
        self.counts = torch.zeros(classes, dtype=torch.int64, device=device)
        self.scale = None

    def loss(self, logits, targets, labels, epoch):
        ramp = max(0.0, min(1.0, (epoch - 8) / 8.0))
        if ramp and self.scale is None:
            raise RuntimeError("Complete epoch-8 calibration before applying expected RCE")
        loss, _, _, p = objective(logits, targets, 0.15, 0.5 * (self.scale or 0.0) * ramp)
        if epoch == 8:
            for row, gradient in zip(self.sums, analytic_gradients(p, targets, 0.15), strict=True):
                row.index_add_(0, labels, gradient.double().square().sum(1))
            self.counts += torch.bincount(labels, minlength=len(self.counts))
        return loss

    def finish_epoch(self, epoch, total_rows):
        if epoch != 8:
            return
        if int(self.counts.sum()) != total_rows or (self.counts == 0).any():
            raise RuntimeError("Calibration requires a complete epoch and every training class")
        rms = (self.sums / self.counts / len(self.counts)).mean(1).sqrt()
        if not torch.isfinite(rms).all() or (rms <= 0).any():
            raise RuntimeError("Cannot calibrate a degenerate CE or expected RCE gradient")
        self.scale = float(rms[0] / rms[1])
