import numpy as np
import pyvista as pv

from pyvistaqt import BackgroundPlotter

def generate_random_box(min_size, max_size):
    """ランダムなサイズの直方体を生成する関数"""
    return np.random.uniform(low=min_size, high=max_size, size=3)


def compute_vertices(position, size):
    """
    直方体の位置とサイズから各頂点の座標を計算する関数
    Args:
        position (numpy.ndarray): 直方体の中心座標を表す numpy 配列 (x, y, z)
        size (numpy.ndarray): 直方体のサイズを表す numpy 配列 (width, height, depth)
    Returns:
        numpy.ndarray: 各頂点の座標を表す numpy 配列 (8, 3)
    """
    # 直方体のサイズの半分を計算
    half_size = size / 2

    # 各頂点の座標を計算
    vertices = np.array([
        [position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2]],  # 前上右
        [position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2]],  # 前下右
        [position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2]],  # 前上左
        [position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2]],  # 前下左
        [position[0] + half_size[0], position[1] + half_size[1], position[2] - half_size[2]],  # 後上右
        [position[0] + half_size[0], position[1] - half_size[1], position[2] - half_size[2]],  # 後下右
        [position[0] - half_size[0], position[1] + half_size[1], position[2] - half_size[2]],  # 後上左
        [position[0] - half_size[0], position[1] - half_size[1], position[2] - half_size[2]]   # 後下左
    ])

    return vertices


def check_collision(box1, box2):
    """
    2つの直方体が当たりしているかどうかを判定する関数
    Args:
        position1 (numpy.ndarray): 1つ目の直方体の位置を表すnumpy配列 (x, y, z)
        size1 (numpy.ndarray): 1つ目の直方体のサイズを表すnumpy配列 (width, height, depth)
        position2 (numpy.ndarray): 2つ目の直方体の位置を表すnumpy配列 (x, y, z)
        size2 (numpy.ndarray): 2つ目の直方体のサイズを表すnumpy配列 (width, height, depth)
    Returns:
        bool: 2つの直方体が当たりしている場合はTrue、そうでない場合はFalse
    """
    # 1つ目の直方体の対角線上の点の座標を計算
    box1_min = box1[0] - box1[1] / 2
    box1_max = box1[0] + box1[1] / 2

    # 2つ目の直方体の対角線上の点の座標を計算
    box2_min = box2[0] - box2[1] / 2
    box2_max = box2[0] + box2[1] / 2

    # 干渉しているかどうかを判定
    collision = np.all(box1_min <= box2_max) and np.all(box1_max >= box2_min)
    # collision = np.any(box1_min <= box2_max) and np.any(box1_max >= box2_min)
    return collision


def is_fully_contained(box1, box2):
    """直方体1が直方体2に完全に含まれるかどうかを判定する関数"""
    # box1の境界ボックスの座標を計算
    box1_min = box1[0] - box1[1] / 2
    box1_max = box1[0] + box1[1] / 2
    # box2の境界ボックスの座標を計算
    box2_min = box2[0] - box2[1] / 2
    box2_max = box2[0] + box2[1] / 2
    # box1の境界ボックスがbox2の内部に含まれるかどうかをチェック
    return np.all(box1_min >= box2_min) and np.all(box1_max <= box2_max)


def random_placement(box_size_range, num_boxes, container_size):
    """直方体をランダムに配置する関数"""
    outer_box = ([0,0,0], container_size)
    containers = []
    for _ in range(num_boxes):
        while True:
            # ランダムなサイズの直方体を生成
            box_size = generate_random_box(*box_size_range)
            # ランダムな座標を生成
            position = np.random.uniform(low=-container_size/2, high=container_size/2, size=3)
            # 直方体の位置とサイズを定義
            box = (position, box_size)
            import ipdb;ipdb.set_trace()

            # 他の直方体と干渉しないかチェック
            if all(not check_collision(box, other) for other in containers) and is_fully_contained(box, outer_box):
            # if all(not check_collision(box, other) for other in containers):
                containers.append(box)
                break
    return containers



def random_placement_const_mode_1(box_arrange_config,container_size):
    """直方体をランダムに配置する関数"""
    outer_box = ([0,0,0], container_size)
    containers = []

    for idx, key in enumerate(box_arrange_config):
        while True:
            # ランダムなサイズの直方体を生成
            box_size = np.random.uniform(box_arrange_config[key]["size"]["min"],box_arrange_config[key]["size"]["max"])
            # ランダムな座標を生成
            position = np.random.uniform(box_arrange_config[key]["position"]["min"],box_arrange_config[key]["position"]["max"])
            # 直方体の位置とサイズを定義
            box = (position, box_size)
            # 他の直方体と干渉しないかチェック
            if all(not check_collision(box, other) for other in containers) and is_fully_contained(box, outer_box):
            # if all(not check_collision(box, other) for other in containers):
                containers.append(box)
                break
            else:
                # print("not match")
                pass
    return containers




if __name__ == '__main__':

    # パラメータの設定
    min_box_size = 0.01  # 直方体の最小サイズ
    max_box_size = 0.3  # 直方体の最大サイズ
    num_boxes = 1  # 配置する直方体の数

    container_size =  np.array([0.1, 0.05, 0.07])-0.005 # 直方体を配置する容器のサイズ

    # 直方体をランダムに配置
    # containers = random_placement((min_box_size, max_box_size), num_boxes, container_size)


    box_arrange_config = {"Box_1":{"position":{"min" :np.asarray([0.0, -0.02, 0.0]),
                                               "max" :np.asarray([0.0, -0.005,0.0])},
                                   "size":{"min":np.asarray([0.08, 0.00, 0.05]),
                                           "max":np.asarray([0.10,  0.01, 0.1])}
                                            },
                          "Box_2":{"position":{"min" :np.asarray([0.01, 0.005, 0.0]),
                                               "max" :np.asarray([0.03, 0.02,0.0])},
                                   "size":{"min":np.asarray([0.03, 0.01, 0.02]),
                                           "max":np.asarray([0.04, 0.02, 0.05])}
                                            },
                          "Box_3":{"position":{"min" :np.asarray([-0.01, 0.0, 0.0]),
                                               "max" :np.asarray([-0.03, 0.02,0.0])},
                                   "size":{"min":np.asarray([0.01, 0.01, 0.01]),
                                           "max":np.asarray([0.03, 0.03, 0.03])}
                                            }
                          }


    box_arrange_config = {"Box_1":{"position":{"min" :np.asarray([0.0, -0.02, 0.0]),
                                               "max" :np.asarray([0.0, -0.005,0.0])},
                                   "size":{"min":np.asarray([0.08, 0.00, 0.05]),
                                           "max":np.asarray([0.10,  0.01, 0.1])}
                                            },
                          "Box_2":{"position":{"min" :np.asarray([-0.01, 0.0, 0.0]),
                                               "max" :np.asarray([-0.03, 0.02,0.0])},
                                   "size":{"min":np.asarray([0.03, 0.01, 0.02]),
                                           "max":np.asarray([0.04, 0.02, 0.05])}
                                            },
                          "Box_3":{"position":{"min" :np.asarray([0.01, 0.005, 0.0]),
                                               "max" :np.asarray([0.03, 0.02,0.0])},
                                   "size":{"min":np.asarray([0.01, 0.01, 0.01]),
                                           "max":np.asarray([0.03, 0.03, 0.03])}
                                            }
                          }

    box_arrange_config = {"Box_1":{"position":{"min" :np.asarray([0.0, -0.04, 0.0]),
                                               "max" :np.asarray([0.0, -0.00,0.0])},
                                   "size":{"min":np.asarray([0.08, 0.00, 0.05]),
                                           "max":np.asarray([0.10,  0.015, 0.1])}
                                            },
                          "Box_2":{"position":{"min" :np.asarray([-0.01, 0.0, 0.0]),
                                               "max" :np.asarray([-0.03, 0.02,0.0])},
                                   "size":{"min":np.asarray([0.03, 0.01, 0.02]),
                                           "max":np.asarray([0.04, 0.02, 0.05])}
                                            },
                          "Box_3":{"position":{"min" :np.asarray([0.01, 0.005, 0.0]),
                                               "max" :np.asarray([0.03, 0.02,0.0])},
                                   "size":{"min":np.asarray([0.01, 0.01, 0.01]),
                                           "max":np.asarray([0.03, 0.03, 0.03])}
                                            }
                          }



    # import ipdb;ipdb.set_trace()


    # containers = random_placement(box_arrange_config, container_size)
    # import ipdb;ipdb.set_trace()


    ####################
    # init plot setting
    ####################
    plotter = BackgroundPlotter(show = True,
            window_size=(1080, 1080), title='Cut Visualization')
    plotter.open_gif("./hogess"+".gif")
    plotter.open_movie("./hoe"+".mp4")



    outer_bounds    = np.asarray([-container_size[0], container_size[0], -container_size[1], container_size[1], -container_size[2], container_size[2]])/2.0
    outer_cubes     = pv.Box(outer_bounds)
    plotter.add_mesh(outer_cubes, opacity=0.1, show_edges=True, color = [0.1,0.1,0.1])

    arrow_x = pv.Arrow(
    start=(0, 0, 0), direction=(1, 0, 0), scale=0.01)
    arrow_y = pv.Arrow(
    start=(0, 0, 0), direction=(0, 1, 0), scale=0.01)
    arrow_z = pv.Arrow(
    start=(0, 0, 0), direction=(0, 0, 1), scale=0.01)
    # plotter.add_camera_orientation_widget()
    plotter.add_mesh(arrow_x, color="r")
    plotter.add_mesh(arrow_y, color='g')
    plotter.add_mesh(arrow_z, color='b')


    inner_box_1 = pv.Box(outer_bounds*0.3)
    inner_box_2 = pv.Box(outer_bounds*0.3)
    inner_box_3 = pv.Box(outer_bounds*0.3)

    inner_box_list =[inner_box_1,inner_box_2,inner_box_3]


    # plotter.add_mesh(inner_box_1, opacity=0.8, color = [0.8,0.2,0,2],show_edges=True,)
    # plotter.add_mesh(inner_box_2, opacity=0.8, color = [0.8,0.8,0,2],show_edges=True,)
    # plotter.add_mesh(inner_box_3, opacity=0.8, color = [0.2,0.8,0,8],show_edges=True,)

    plotter.add_mesh(inner_box_1, opacity=0.8, color = [0.2,0.8,0,8],show_edges=True,)
    plotter.add_mesh(inner_box_2, opacity=0.8, color = [0.8,0.2,0,2],show_edges=True,)
    plotter.add_mesh(inner_box_3, opacity=0.8, color = [0.8,0.8,0,2],show_edges=True,)


    plotter.set_background('white')
    plotter.render()
    plotter.write_frame()
    plotter.app.processEvents()

    for j in range(100):

        # 直方体をランダムに配置
        # containers = random_placement((min_box_size, max_box_size), num_boxes, container_size)
        containers = random_placement_const_mode_1(box_arrange_config, container_size)


        for i, container in enumerate(containers):
            print(f"Box {i+1}: position={container[0]}, size={container[1]}")
            box_size = container[1]/2.0
            box_pos  = container[0]

            outer_bounds = [-box_size[0], box_size[0], -box_size[1], box_size[1], -box_size[2], box_size[2]]
            inner_cube_ = pv.Box(outer_bounds)
            inner_cube  = inner_cube_.translate(box_pos)
            # inner_box_list.append(inner_cube)
            inner_box_list[i].points = inner_cube.points
            inner_box_list[i].faces = inner_cube.faces
        plotter.render()
        plotter.app.processEvents()
        plotter.write_frame()
        import time;time.sleep(.05)
        # plotter.show()

