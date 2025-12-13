# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import importlib
import json
from typing import Any, Dict, Optional, Protocol, Tuple, cast

import torch
from torch import optim as optim


def _import_optimizer(module: str, attr: str) -> Any:
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        return None


Adafactor = cast(Any, _import_optimizer("timm.optim.adafactor", "Adafactor"))
Adahessian = cast(Any, _import_optimizer("timm.optim.adahessian", "Adahessian"))
AdamP = cast(Any, _import_optimizer("timm.optim.adamp", "AdamP"))
Lookahead = cast(Any, _import_optimizer("timm.optim.lookahead", "Lookahead"))
Nadam = cast(Any, _import_optimizer("timm.optim.nadam", "Nadam"))
NvNovoGrad = cast(Any, _import_optimizer("timm.optim.nvnovograd", "NvNovoGrad"))
RAdam = cast(Any, _import_optimizer("timm.optim.radam", "RAdam"))
RMSpropTF = cast(Any, _import_optimizer("timm.optim.rmsprop_tf", "RMSpropTF"))
SGDP = cast(Any, _import_optimizer("timm.optim.sgdp", "SGDP"))


class OptimizerArgs(Protocol):
    opt: str
    weight_decay: float
    lr: float
    momentum: float
    opt_eps: Optional[float]
    opt_betas: Optional[Tuple[float, float]]

FusedNovoGrad: Any = None
FusedAdam: Any = None
FusedLAMB: Any = None
FusedSGD: Any = None
has_apex = False

try:
    apex_module = importlib.import_module("apex.optimizers")
except ImportError:
    apex_module = None

if apex_module is not None:
    FusedNovoGrad = getattr(apex_module, "FusedNovoGrad", None)
    FusedAdam = getattr(apex_module, "FusedAdam", None)
    FusedLAMB = getattr(apex_module, "FusedLAMB", None)
    FusedSGD = getattr(apex_module, "FusedSGD", None)
    has_apex = all((FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD))


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
    args: OptimizerArgs,
    model,
    get_num_layer=None,
    get_layer_scale=None,
    filter_bias_and_bn=True,
    skip_list=None,
    **kwargs,
):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args: Dict[str, Any] = {"lr": args.lr, "weight_decay": weight_decay}
    opt_eps = getattr(args, "opt_eps", None)
    if opt_eps is not None:
        opt_args["eps"] = opt_eps
    opt_betas = getattr(args, "opt_betas", None)
    if opt_betas is not None:
        opt_args["betas"] = opt_betas
    
    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        if Nadam is None:
            raise ImportError("Nadam optimizer is unavailable (timm package missing)")
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        if RAdam is None:
            raise ImportError("RAdam optimizer is unavailable (timm package missing)")
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        if AdamP is None:
            raise ImportError("AdamP optimizer is unavailable (timm package missing)")
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        if Adafactor is None:
            raise ImportError("Adafactor optimizer is unavailable (timm package missing)")
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        if Adahessian is None:
            raise ImportError("Adahessian optimizer is unavailable (timm package missing)")
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        if RMSpropTF is None:
            raise ImportError("RMSpropTF optimizer is unavailable (timm package missing)")
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        if NvNovoGrad is None:
            raise ImportError("NvNovoGrad optimizer is unavailable (timm package missing)")
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
