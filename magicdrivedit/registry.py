# 经典注册机制的代码

from copy import deepcopy # 这个函数可以深拷贝一个对象，也就是说不仅复制对象本身，还复制它内部嵌套的所有对象，互不影响

import torch.nn as nn
from mmengine.registry import Registry # 用于模块构建的注册机制（类似插件管理器）


def build_module(module, builder, **kwargs):
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build. 可以是一个字典（表示模块的配置）、一个已经构建好的 nn.Module 对象，或者 None
        builder (Registry): The registry to build module.一个 Registry 实例，它知道如何根据 module 中的配置来构建模块
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if module is None:
        return None
    if isinstance(module, dict):
        cfg = deepcopy(module)
        for k, v in kwargs.items(): # kwargs.items()是空的
            cfg[k] = v
        return builder.build(cfg) # 使用注册器 builder 的 .build(cfg) 方法，根据配置字典构建一个模块（如一个模型）
    elif isinstance(module, nn.Module):
        return module
    else:
        raise TypeError(f"Only support dict and nn.Module, but got {type(module)}.")

# @MODELS.register_module()
MODELS = Registry(
    "model", # 创建一个名为 "model" 的注册器 MODELS
    locations=["magicdrivedit.models"], # 表示可以在这个 Python 包路径下自动查找所有可以注册的模块（比如模型类）
)

SCHEDULERS = Registry(
    "scheduler",
    locations=["magicdrivedit.schedulers"],
)

DATASETS = Registry(
    "dataset",
    locations=["magicdrivedit.datasets"],
)
