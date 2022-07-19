# TODO: Jonas doc strings
import os
import imageio
import cv2
import numpy as np


def image_functionality(func):
    """
    Decorator to allow for logging functionality.
    Should be added to all plotting functions of visualizer.
    The plot function has to return a np.uint8
    @image_functionality
    def plot_segmentation(self, seg, **kwargs):
        return np.zeros((H, W, 3), dtype=np.uint8)
    not_log [optional, bool, default: false] : the decorator is ignored if set to true
    epoch [optional, int, default: visualizer.epoch ] : overwrites visualizer, epoch used to log the image
    store [optional, bool, default: visualizer.store ] : overwrites visualizer, flag if image is stored to disk
    tag [optinal, str, default: tag_not_defined ] : stores image as: visulaiter._p_visu{epoch}_{tag}.png
    """

    def wrap(*args, **kwargs):
        img = func(*args, **kwargs)

        if not kwargs.get("not_log", False):
            log_exp = args[0]._pl_model.logger is not None
            tag = kwargs.get("tag", "tag_not_defined")

            if kwargs.get("store", None) is not None:
                store = kwargs["store"]
            else:
                store = args[0]._store

            if kwargs.get("epoch", None) is not None:
                epoch = kwargs["epoch"]
            else:
                epoch = args[0]._epoch

            # Store to disk
            if store:
                p = os.path.join(args[0]._p_visu, f"{epoch}_{tag}.png")
                imageio.imwrite(p, img)

            if log_exp:
                if args[0]._pl_model.logger is not None:
                    H, W, C = img.shape
                    ds = cv2.resize(img, dsize=(int(W / 2), int(H / 2)), interpolation=cv2.INTER_CUBIC)
                    try:
                        # logger == neptuneai
                        from neptune.new.types import File

                        args[0]._pl_model.logger.experiment[tag].log(File.as_image(np.float32(ds) / 255), step=epoch)
                    except:
                        try:
                            # logger == wandb
                            import wandb

                            args[0]._pl_model.logger.experiment.log({tag: [wandb.Image(ds, caption=tag)]}, commit=True)
                        except:
                            try:
                                # logger == tensorboard
                                args[0]._pl_model.logger.experiment.add_image(
                                    tag=tag,
                                    img_tensor=ds,
                                    global_step=epoch,
                                    dataformats="HWC",
                                )
                            except:
                                print("Tensorboard Logging and Neptune Logging failed !!!")
                                pass

        return func(*args, **kwargs)

    return wrap
