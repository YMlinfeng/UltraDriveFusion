# import pickle
# import pprint

# def main():
#     # pickle文件路径
#     pkl_file = "/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"
    
#     # 以二进制读模式加载pickle文件
#     with open(pkl_file, "rb") as f:
#         pkl = pickle.load(f)

#     representative_data = {}

#     # metadata：保留全部数据（注意：键名为"metadata"）
#     representative_data["metadata"] = pkl.get("metadata")

#     # scene_tokens：只选取前100和后100，保存为字典格式
#     if "scene_tokens" in pkl:
#         tokens = pkl["scene_tokens"]
#         if isinstance(tokens, list) and len(tokens) >= 2:
#             representative_data["scene_tokens"] = {
#                 "head": tokens[:1],
#                 "tail": tokens[-1:]
#             }
#         else:
#             representative_data["scene_tokens"] = tokens

#     # infos：只选取前3和后3，这里每个元素本身也是一个字典
#     if "infos" in pkl:
#         infos = pkl["infos"]
#         if isinstance(infos, list) and len(infos) >= 2:
#             representative_data["infos"] = {
#                 "head": infos[:1],
#                 "tail": infos[-1:]
#             }
#         else:
#             representative_data["infos"] = infos

#     # 生成格式化字符串，便于阅读
#     formatted_content = pprint.pformat(representative_data, indent=2, width=120)

#     # 写入到TXT文件中
#     output_file = "pkl_representative_preview.txt"
#     with open(output_file, "w", encoding="utf-8") as out_f:
#         out_f.write(formatted_content)

#     print(f"代表性数据已成功写入 {output_file}")

# if __name__ == "__main__":
#     main()




import pickle
import pprint

def extract_info_sample(info_record):
    """
    从单条info记录中提取关键信息：
      - 整条记录的时间戳（键 'timestamp'）作为 sample_timestamp
      - info_record["cams"] 中，每个摄像头的图片路径（data_path）和采集时间（timestamp）
    """
    result = {}

    # 提取整体记录的时间戳
    result['sample_timestamp'] = info_record.get('timestamp', None)

    # 提取cams里每个摄像头的关注信息
    cams = info_record.get('cams', {})
    cams_extracted = {}
    for cam_key, cam_data in cams.items():
        if isinstance(cam_data, dict):
            extracted = {}
            extracted['data_path'] = cam_data.get('data_path', None)
            extracted['timestamp'] = cam_data.get('timestamp', None)
            cams_extracted[cam_key] = extracted
    result['cams'] = cams_extracted

    return result

def main():
    # pickle文件路径
    pkl_file = "/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"
    
    # 以二进制读取pickle文件
    with open(pkl_file, "rb") as f:
        pkl = pickle.load(f)
    
    representative_data = {}
    
    # 1. metadata：直接保存所有内容（通常较小）
    representative_data["metadata"] = pkl.get("metadata")
    
    # 2. scene_tokens：只保存列表的前100个和后100个
    tokens = pkl.get("scene_tokens")
    if isinstance(tokens, list) and len(tokens) >= 2:
        representative_data["scene_tokens"] = {
            "head": tokens[:1],
            "tail": tokens[-1:]
        }
    else:
        representative_data["scene_tokens"] = tokens
    
    # 3. infos：原来可能有很多条，这里只保留前 3 条和后 3 条
    infos = pkl.get("infos")
    if isinstance(infos, list) and len(infos) >= 2:
        head_infos = [extract_info_sample(info) for info in infos[:1]]
        tail_infos = [extract_info_sample(info) for info in infos[-1:]]
        representative_data["infos"] = {
            "head": head_infos,
            "tail": tail_infos
        }
    else:
        representative_data["infos"] = infos
    
    # 生成易读的格式化字符串，写入TXT文件
    formatted_content = pprint.pformat(representative_data, indent=2, width=120)
    output_file = "pkl_representative_preview_useful.txt"
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(formatted_content)
    
    print(f"代表性数据已成功写入 {output_file}")
    
    # ----------------------------------------------
    # 以下给出如何利用这些时间戳构建顺序排列的方法：
    #
    # 假设你希望按照每条info记录的整体时间戳（sample_timestamp）来排序：
    #   sorted_infos = sorted(infos, key=lambda info: info.get('timestamp', 0))
    # 其中，info.get('timestamp', 0) 取得每个info记录的顶级'timestamp'字段。
    #
    # 然后你可以提取每条info中所有摄像头的图片路径，并按总体时间顺序保存或进一步处理。
    #
    # 例如：
    #
    # sorted_infos = sorted(infos, key=lambda info: info.get('timestamp', 0))
    # for info in sorted_infos:
    #     overall_ts = info.get('timestamp')
    #     cams = info.get('cams', {})
    #     for cam, cam_info in cams.items():
    #         img_path = cam_info.get('data_path')
    #         cam_ts = cam_info.get('timestamp')
    #         # 这里可以将 img_path 保存到列表中，或做其他处理
    #         print(f"Info整体时间戳：{overall_ts}, 摄像头：{cam}, 图片路径：{img_path}, CAM时间戳：{cam_ts}")
    #
    # 如果你发现每条info内部各CAM的时间戳有轻微偏差，可以选择整体采样记录的'timestamp'作为排序依据，
    # 或者用各个摄像头的时间戳（如果需要更精确的同步）作为排序条件。
    # ----------------------------------------------

if __name__ == "__main__":
    main()