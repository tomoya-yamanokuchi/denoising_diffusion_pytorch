import os
import pyvista as pv


def get_benchmark_model(dataset_path, model_config, for_data_gen = False):


        if  for_data_gen is True:
            for idx, val in enumerate(model_config["outer_parts"]):
                    print(val)
                    mesh_path   =  os.path.normpath(dataset_path+model_config["outer_parts"][val]["path"])
                    mesh        = pv.read(mesh_path)
                    mesh_color  = model_config["outer_parts"][val]["color"]
                    model_config["outer_parts"][val].update({"mesh":mesh})
                    print(f"load: mesh_path | {mesh_path}, color | {mesh_color}")

            for idx, val in enumerate(model_config["internal_parts"]):
                    print(val)
                    mesh_path   =  os.path.normpath(dataset_path+model_config["internal_parts"][val]["path"])
                    mesh        = pv.read(mesh_path)
                    mesh_color  = model_config["internal_parts"][val]["color"]
                    model_config["internal_parts"][val].update({"mesh":mesh})
                    print(f"load: mesh_path | {mesh_path}, color | {mesh_color}")

            return model_config

        else:
            mesh_components = {}
        for idx, val in enumerate(model_config["outer_parts"]):
                print(val)
                mesh_path   =  os.path.normpath(dataset_path+model_config["outer_parts"][val]["path"])
                mesh        = pv.read(mesh_path)
                mesh_color  = model_config["outer_parts"][val]["color"]
                data        = {val:{    "mesh":mesh,
                                        "color":mesh_color}}
                print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                mesh_components.update(data)


        for idx, val in enumerate(model_config["internal_parts"]):
                print(val)
                mesh_path   =  os.path.normpath(dataset_path+model_config["internal_parts"][val]["path"])
                mesh        = pv.read(mesh_path)
                mesh_color  = model_config["internal_parts"][val]["color"]
                data        = {val:{    "mesh":mesh,
                                        "color":mesh_color}}
                print(f"load: mesh_path | {mesh_path}, color | {data[val]['color']}")
                mesh_components.update(data)

        return mesh_components