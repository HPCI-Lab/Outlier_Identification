

def compute_gradient_norm_per_layer(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute the L2 norm of the gradients for this parameter
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[name] = grad_norm
    return grad_norms

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def compute_gradient_norm_deepspeed(model): 
    total_norm = 0.0
    from deepspeed.utils import safe_get_local_grad
    for _, lp in model.named_parameters(): 
        if lp.grad is not None: 
            hp_grad = safe_get_local_grad(lp)
            hp_grad = sum(hp_grad)
            total_norm += hp_grad.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

