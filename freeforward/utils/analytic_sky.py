"""Algebraic evaluation of the existing 3->512->3 conditional sky MLP.

No parameters, inputs or nonlinearities are removed. LayerNorm statistics
are evaluated through the 3x3 covariance of the first linear map instead
of materializing a 512-channel image. FP32 equivalence and AMP tolerances
must be validated before this optional execution path is used.
"""
import torch

def analytic_sky(head, directions, condition):
    if head.linear.in_features != 3 or head.norm.elementwise_affine:
        raise ValueError('Analytic sky requires3D input and non-affine LayerNorm')
    projected = head.condition_mapping(condition.squeeze(1))
    shift, scale = head.adaLN_modulation(projected).chunk(2,dim=-1)
    output_dtype = shift.dtype
    # The small condition MLP retains the original AMP behavior. The analytic
    # LayerNorm and effective matrices use FP32 (FP64 in gradient tests), so
    # numerical cancellation in the covariance is not amplified by BF16.
    dtype = torch.float64 if head.linear.weight.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=directions.device.type, enabled=False):
        x = directions.to(dtype)
        shape = x.shape
        x = x.reshape(shape[0],-1,3)
        weight = head.linear.weight.to(dtype)
        bias = head.linear.bias.to(dtype)
        output_weight = head.output.weight.to(dtype)
        output_bias = head.output.bias.to(dtype)
        centered_weight = weight-weight.mean(dim=0,keepdim=True)
        centered_bias = bias-bias.mean()
        hidden = weight.shape[0]
        covariance = centered_weight.T@centered_weight/hidden
        cross = centered_weight.T@centered_bias/hidden
        variance_bias = centered_bias.square().mean()
        variance = ((x@covariance)*x).sum(-1)+2*(x@cross)+variance_bias
        inverse_std = torch.rsqrt(variance.clamp_min(0)+head.norm.eps)
        conditioned_output = output_weight[None]*(1+scale.to(dtype))[:,None,:]
        effective_weight = conditioned_output@centered_weight
        effective_bias = (conditioned_output*centered_bias).sum(-1)
        modulation_bias = shift.to(dtype)@output_weight.T+output_bias
        y = (torch.bmm(x,effective_weight.transpose(1,2))+effective_bias[:,None])*inverse_std[...,None]
        y = y+modulation_bias[:,None]
        return y.reshape(*shape[:-1],head.output.out_features).to(output_dtype)

def install_analytic_sky(model):
    # Replace a callable, not the Module: state_dict names and all original
    # trainable Parameters remain identical, including strict checkpoint load.
    head = model.sky_head
    if getattr(head,'_analytic_sky_installed',False):
        raise RuntimeError('Analytic sky already installed')
    original = head.forward
    head.forward = lambda directions, condition: analytic_sky(head,directions,condition)
    head._analytic_sky_installed = True
    return original
