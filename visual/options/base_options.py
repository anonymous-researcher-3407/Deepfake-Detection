import argparse


class BaseOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-d",
            "--data-root",
            type=str,
            help="data root",
        )
        self.parser.add_argument(
            "-o",
            "--cache-dir",
            type=str,
            default="cache/",
            help="cache dir",
        )
        self.parser.add_argument(
            "-fn",
            "--visual-cache-file-name",
            type=str,
            default="visual_cache.zarr",
            help="visual cache file name",
        )
				self.parser.add_argument(
            "-fn",
            "--audio-cache-file-name",
            type=str,
            default="audio_cache.zarr",
            help="audio cache file name",
        )
				self.parser.add_argument(
            "-fn",
            "--fusion-cache-file-name",
            type=str,
            default="fusion_cache.zarr",
            help="fusion cache file name",
        )
        self.parser.add_argument(
            "--vqvae-ckpt",
            type=str,
            default="weights/vqvae.pt",
            help="checkpoint path for vqvae",
        )
        self.parser.add_argument(
            "--lmm-ckpt",
            type=str,
            default="sparklexfantasy/llava-7b-1.5-rfrd",
            help="the checkpoint of lmm",
        )
        self.parser.add_argument(
            "--lmm-base", type=str, default=None, help="the base model of lmm"
        )
        self.parser.add_argument(
            "--st-ckpt",
            type=str,
            default="weights/vit.pth",
            help="the checkpoint of the pretrained checkpoint of hybrid vit in ST branch",
        )
        self.parser.add_argument(
            "--st-pretrained",
            type=bool,
            default=True,
            help="whether to use the pretrained checkpoint of hybrid vit in ST branch",
        )
        self.parser.add_argument(
            "--model-name", type=str, default="MMDet", help="the model name"
        )
        self.parser.add_argument(
            "--expt", type=str, default="MMDet_01", help="the experiment name"
        )
        self.parser.add_argument(
            "--window-size", type=int, default=10, help="window size for video clips"
        )
        self.parser.add_argument(
            "--new-tokens",
            type=int,
            default=64,
            help="the number of extracted tokens of the output layer of lmm",
        )
        self.parser.add_argument(
            "--selected-layers",
            type=int,
            nargs="+",
            default=[-1],
            help="the selected layers for feature of lmm",
        )
        self.parser.add_argument(
            "--interval",
            type=int,
            default=200,
            help="the interval between cached mm representataions of lmm, only available for caching",
        )
        self.parser.add_argument(
            "--load-8bit", action="store_true", help="whether load lmm of 8 bit"
        )
        self.parser.add_argument(
            "--load-4bit", action="store_true", help="whether load lmm of 4 bit"
        )
        self.parser.add_argument("--seed", type=int, default=1, help="random seed")
        self.parser.add_argument("--gpus", type=int, default=1, help="number for gpus")
        self.parser.add_argument(
            "--cache-mm",
            action="store_true",
            help="whether load mm encoder or use cached representations",
        )
        self.parser.add_argument("--debug", action="store_true", help="debug mode")
        self.parser.add_argument(
            "--batch-size", type=int, default=1, help="batch size for testing"
        )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="number for dataloader workers per gpu",
        )
        self.parser.add_argument(
            "--visual-logits",
            type=str,
						default="predict",
        )
				self.parser.add_argument(
            "--dataset",
            type=str,
        )

    def parse(self):
        return self.parser.parse_args()
