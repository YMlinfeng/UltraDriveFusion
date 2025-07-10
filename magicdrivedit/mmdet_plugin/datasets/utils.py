import mmcv


def extract_result_dict(results, key):
    """提取并返回结果字典中对应 key 的数据。

    ``results`` 是从 `pipeline(input_dict)` 输出的字典，
        它是从 ``Dataset`` 类中加载的数据。
    其中的数据项可能被包装在 list、tuple 或 DataContainer 中，
        因此此函数的主要作用是从这些封装中提取数据。

    参数:
        results (dict): 使用 pipeline 加载的数据。
        key (str): 所需数据的键。

    返回:
        np.ndarray | torch.Tensor | None: 数据项。
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data
