from denoising_diffusion_pytorch.env.voxel_cut_sim_v1 import voxel_cut_handler


class EnsembleImageBuilder:
    def __init__(self, obs_model: voxel_cut_handler):
        self.obs_model = obs_model

    def build_from_generated_samples(self, last_step_images):
        mean_image_z = last_step_images.mean(0) / 255.0
        return self.build_from_image_z(mean_image_z)

    def build_from_image_z(self, image_z):
        self.obs_model.cast_2d_image_to_box_color(
            img    = image_z,
            config = {"axis": "z"},
        )
        return {
            "z": self.obs_model.get_2d_image(axis="z"),
            "x": self.obs_model.get_2d_image(axis="x"),
            "y": self.obs_model.get_2d_image(axis="y"),
        }
