
import numpy as np
import torch
from torch.autograd import Variable
from visdom import Visdom

def gray2rgb_norm(gray_image):
    min_, max_ = gray_image.min(), gray_image.max()
    gray_image = (gray_image - min_) / (max_ - min_)
    return torch.cat([gray_image, gray_image, gray_image], 0)

def gray2rgb(gray_image):
    return torch.cat([gray_image, gray_image, gray_image], 0)


class Dashboard:
    def __init__(self, port, env_name="GrayChannel"):
        self.vis = Visdom(port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]),
                                    Y=np.array([y, y]), env=self.env, opts=dict(
                                    legend=[split_name],
                                    title=var_name,
                                    xlabel="Iters",
                                    ylabel=var_name
                                    ))
        else:
            self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
                                win=self.plots[var_name], name=split_name,
                                update='append'
                         )
            
    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            imgae = image.data
        image = image.numpy()
        img_env = self.env + '_images'
        self.vis.image(image, env=img_env, opts=dict(title=title))
