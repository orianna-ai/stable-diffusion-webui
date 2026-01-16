import os

from modules import modelloader, errors
from modules.shared import cmd_opts, opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerDAT(Upscaler):
    def __init__(self, user_path):
        self.name = "DAT"
        self.user_path = user_path
        self.scalers = []
        super().__init__()

        for file in self.find_models(ext_filter=[".pt", ".pth"]):
            name = modelloader.friendly_name(file)
            scaler_data = UpscalerData(name, file, upscaler=self, scale=None)
            self.scalers.append(scaler_data)

        for model in get_dat_models(self):
            if model.name in opts.dat_enabled_models:
                self.scalers.append(model)

    def do_upscale(self, img, path):
        try:
            info = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load DAT model {path}", exc_info=True)
            return img

        model_descriptor = modelloader.load_spandrel_model(
            info.local_data_path,
            device=self.device,
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture="DAT",
        )
        return upscale_with_model(
            model_descriptor,
            img,
            tile_size=opts.DAT_tile,
            tile_overlap=opts.DAT_tile_overlap,
        )

    def load_model(self, path):
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"DAT data missing: {scaler.local_data_path}")
                return scaler
        raise ValueError(f"Unable to find model info: {path}")


def get_dat_models(scaler):
    """
    Returns all available DAT upscaler models.
    Models are pre-downloaded to /app/models/DAT during Docker build.
    
    Variants:
    - DAT: Original full model (~14.8M params)
    - DAT-2: Balanced performance (~11.2M params)
    - DAT-S: Small variant (~11.2M params, slightly lighter)
    - DAT-light: Lightweight for limited compute (~573K params)
    """
    dat_models_path = "/app/models/DAT"
    
    return [
        # DAT (original) models
        UpscalerData(
            name="DAT x2",
            path=os.path.join(dat_models_path, "DAT_x2.pth"),
            scale=2,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT x3",
            path=os.path.join(dat_models_path, "DAT_x3.pth"),
            scale=3,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT x4",
            path=os.path.join(dat_models_path, "DAT_x4.pth"),
            scale=4,
            upscaler=scaler,
        ),
        # DAT-2 models (balanced performance)
        UpscalerData(
            name="DAT-2 x2",
            path=os.path.join(dat_models_path, "DAT_2_x2.pth"),
            scale=2,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT-2 x3",
            path=os.path.join(dat_models_path, "DAT_2_x3.pth"),
            scale=3,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT-2 x4",
            path=os.path.join(dat_models_path, "DAT_2_x4.pth"),
            scale=4,
            upscaler=scaler,
        ),
        # DAT-S models (small variant)
        UpscalerData(
            name="DAT-S x2",
            path=os.path.join(dat_models_path, "DAT_S_x2.pth"),
            scale=2,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT-S x3",
            path=os.path.join(dat_models_path, "DAT_S_x3.pth"),
            scale=3,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT-S x4",
            path=os.path.join(dat_models_path, "DAT_S_x4.pth"),
            scale=4,
            upscaler=scaler,
        ),
        # DAT-light models (fast, lightweight)
        UpscalerData(
            name="DAT-light x2",
            path=os.path.join(dat_models_path, "DAT_light_x2.pth"),
            scale=2,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT-light x3",
            path=os.path.join(dat_models_path, "DAT_light_x3.pth"),
            scale=3,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT-light x4",
            path=os.path.join(dat_models_path, "DAT_light_x4.pth"),
            scale=4,
            upscaler=scaler,
        ),
    ]
