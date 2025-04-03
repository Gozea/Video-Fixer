import argparse
import restoration

from torchvision.models import mobilenet_v3_small, resnet18

models = {
        "mobilenet_v3_small": mobilenet_v3_small,
        "resnet18": resnet18
        }

if __name__ == "__main__":
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="path to video input")
    parser.add_argument("-m", "--model", type=str, required=False, choices=models.keys(), default="mobilenet_v3_small", help="model to use [mobilenet_v3_small, resnet18]")
    args = parser.parse_args()

    # choose model
    model = models[args.model]

    # video fixer call
    restoration.VideoFixer(args.input, model).fix_video()
