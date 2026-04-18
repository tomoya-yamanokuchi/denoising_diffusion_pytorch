
import pickle
import yaml

import numpy as np

def to_python_type(obj):
    """再帰的に NumPy スカラーや配列をネイティブ Python 型に変換"""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_type(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_action_table(grid_config):
    """_summary_
        define slice action index

    Args:
        grid_config (dict)
    Returns:
        action table (dict): {i:{"axis":data_order[val],"loc":j}})
        i    : Serial number of the action index
        axis : axis name
        loc  : slice index
        In the current configuration, Data_order is unified as [“Z”, “X”, “Y”].
    """


    """Creates an action table that maps action indices to slice operations. In the current configuration, Data_order is unified as [“z”, “x”, “y”].

    Args:
        grid_config (dict): Configuration dictionary for the voxel grid.

    Returns:
        dict: A table mapping action indices to action descriptions.
            Each action includes the axis (e.g., "z", "x", "y") and the slice location.

    Examples:
        >>> action table (dict): {i:{"axis":data_order[val],"loc":j}})
        >>> i    : Serial number of the action index
        >>> axis : axis name
        >>> loc  : slice index
    """

    image_length = grid_config["side_length"]
    action_table  = {}

    i   = 0
    data_order = ["z","x","y"]
    for val in range(len(data_order)):
        for j in range(image_length):
            action_table.update({i:{"axis":data_order[val],"loc":j}})
            i+=1

    return action_table


def convert_action_idx_to_plane(action_idx, action_table, action_pos_candidate):
    act = action_idx
    axis = action_table[act]['axis']
    loc_idx = action_table[act]['loc']
    pos = action_pos_candidate[loc_idx]

    if axis == 'z':
        translation = np.array([0, 0, pos])
        rotation    = np.array([0, 0, 0])
    elif axis == 'y':
        translation = np.array([0, pos, 0])
        rotation    = np.array([90, 0, 0])
    elif axis == 'x':
        translation = np.array([pos, 0, 0])
        rotation    = np.array([0, 90, 0])
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return {
            'axis': axis,
            'position': float(pos),
            'translation': translation.tolist(),
            'rotation': rotation.tolist(),
            'action': act,
            'loc_index': loc_idx,}


def get_cutting_plane(action_data,s_grid_config):


    data_name = action_data
    action_table  = get_action_table(s_grid_config)

    try:
        with open(data_name, 'rb') as f:
            load_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"can not read: {e}")



    intermediate_action  = load_data["intermediate_actions"]
    action_pos_candidate = np.linspace(s_grid_config["bounds"][0],s_grid_config["bounds"][1],s_grid_config["side_length"])


    cutting_plane_config = {}
    for k in range(len(intermediate_action)):
        if 1<len(intermediate_action[k]):
            start = convert_action_idx_to_plane(intermediate_action[k][0], action_table, action_pos_candidate)
            end   = convert_action_idx_to_plane(intermediate_action[k][-1], action_table, action_pos_candidate)
        elif intermediate_action[k]==[0]:
            start = {}
            end   = {}
        else:
            import ipdb;ipdb.set_trace()
            start = convert_action_idx_to_plane(intermediate_action[k][0]+1, action_table, action_pos_candidate)
            end   = convert_action_idx_to_plane(intermediate_action[k][0], action_table, action_pos_candidate)

        cutting_plane_config[k] = {
            'start_position': start,
            'end_position'  : end,
        }

    return cutting_plane_config

if __name__ == '__main__':

    # data_folder = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v14_1/epsilon_greedy_00/Object_6/episode_0/"
    # data_folder = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_conditional_diffusion_a123456_clip_ucb_raw_0.5_v14_1/oracle_obs/Object_6/episode_5/"
    data_folder = "/home/haxhi/workspace/denoising_diffusion_pytorch/logs/Image_diffusion_2D/diffusion_plans/real_model/real_models_dataset_v2_18/dataset_SheetSander_024_eval3/PT100000_B32_T8_partial_obs_vaeac_a123456_clip_ucb_raw_0.5_v13_2/epsilon_greedy_00/Object_6/episode_0/"
    

    data   =  f"{data_folder}/visualization_data.pickle"
    dim_3D = 49
    s_grid_config = {"bounds":(-0.3,0.3,-0.3,0.3,-0.3,0.3), 'side_length':dim_3D}
    cutting_plane_config = get_cutting_plane(action_data = data, s_grid_config = s_grid_config)

    # YAMLに保存
    with open(f"{data_folder}/cutting_plane_config.yaml", "w") as f:
        yaml.dump(to_python_type(cutting_plane_config), f, default_flow_style=False, sort_keys=False)


    # import ipdb;ipdb.set_trace()

