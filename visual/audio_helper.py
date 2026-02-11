import pickle
import zarr
from tqdm import tqdm
import numpy as np
import os
from options.base_options import BaseOption

def main(args, audio_resolution):
		with open(f"/path/to/audio_{audio_resolution}.pkl", "rb") as f:
				data = pickle.load(f)

		file = zarr.open_group(os.path.join(args["cache_dir"], args["audio_cache_file_name"]), mode="a")

		for i in tqdm(data):
				file.array(
						name=i[0][0] + "/" + audio_resolution,
						data=i[:, 1:3].astype(np.float32),
						overwrite=True,
				)

if __name__ == "__main__":
    opt = BaseOption()
    args = opt.parse().__dict__

		audio_resolution = # 160, 320, 640

    main(args, audio_resolution)
