import math
import numpy as np
from PIL import Image


class Mobius_mask(object):
    def __init__(self, dataset, mask_length=10):
        if dataset in ('cifar10', 'cifar100', 'svhn', 'imagenet'):
            self.h = 32
            self.w = 32
        elif dataset == 'tiny':
            self.h = 64
            self.w = 64
        elif dataset == 'stl10':
            self.h = 96
            self.w = 96
        elif dataset == 'pet':
            self.h = 224
            self.w = 224
        elif dataset == 'kmnist':
            self.h = 28
            self.w = 28
        else:
            self.h = 32
            self.w = 32
        self.mask_length = mask_length

        # Precompute coordinate grid (same structure as Mobius class)
        e = [complex(0, 0)] * self.h * self.w
        self.z = np.array(e).reshape(self.h, self.w)
        for i in range(self.h):
            for j in range(self.w):
                self.z[i, j] = complex(i, j)

    def __call__(self, image):
        img_array = np.array(image)
        height, width = self.h, self.w

        a, b, c, d = self._getabcd(height, width)

        # Compute where each original pixel maps to in transformed space
        w = (a * self.z + b) / (c * self.z + d)
        first = np.real(w).astype(int)
        second = np.imag(w).astype(int)

        # Random rectangular mask position in transformed coordinate space
        mask_y = np.random.randint(0, max(1, height - self.mask_length))
        mask_x = np.random.randint(0, max(1, width - self.mask_length))

        # Zero out original pixels that map into the mask region
        in_mask = (
            (first >= mask_y) & (first < mask_y + self.mask_length) &
            (second >= mask_x) & (second < mask_x + self.mask_length)
        )
        img_out = img_array.copy()
        img_out[in_mask] = 0

        return Image.fromarray(img_out.astype(np.uint8))

    def _getabcd(self, height, width):
        raffle = np.random.randint(8)

        if raffle == 0:
            zp = [complex(1, 0.5*width), complex(0.5*height, 0.8*width), complex(0.6*height, 0.5*width)]
            wa = [complex(0.5*height, width-1), complex(0.5*height+0.4*width, 0.5*width), complex(0.5*height, 0.5*width-0.1*height)]
        elif raffle == 1:
            zp = [complex(1, 0.5*width), complex(0.5*height, 0.8*width), complex(0.6*height, 0.5*width)]
            wa = [complex(0.5*height, width-1),
                  complex(0.5*height + 0.3*width*math.sin(0.5*math.pi*0.8), 0.5*width + 0.3*width*math.cos(0.5*math.pi*0.8)),
                  complex(0.5*height + 0.1*height*math.cos(0.5*math.pi*0.2), 0.5*width - 0.1*height*math.sin(0.5*math.pi*0.2))]
        elif raffle == 2:
            zp = [complex(0.3*height, 0.5*width), complex(0.5*height, 0.7*width), complex(0.7*height, 0.5*width)]
            wa = [complex(0.2*height, 0.5*width), complex(0.5*height, 0.8*width), complex(0.8*height, 0.5*width)]
        elif raffle == 3:
            zp = [complex(0.3*height, 0.3*width), complex(0.6*height, 0.8*width), complex(0.7*height, 0.3*width)]
            wa = [complex(0.2*height, 0.3*width), complex(0.6*height, 0.9*width), complex(0.8*height, 0.2*width)]
        elif raffle == 4:
            wa = [complex(1, 0.5*width), complex(0.5*height, 0.8*width), complex(0.6*height, 0.5*width)]
            zp = [complex(0.5*height, width-1), complex(0.5*height+0.4*width, 0.5*width), complex(0.5*height, 0.5*width-0.1*height)]
        elif raffle == 5:
            wa = [complex(1, 0.5*width), complex(0.5*height, 0.8*width), complex(0.6*height, 0.5*width)]
            zp = [complex(0.5*height, width-1),
                  complex(0.5*height + 0.3*width*math.sin(0.5*math.pi*0.8), 0.5*width + 0.3*width*math.cos(0.5*math.pi*0.8)),
                  complex(0.5*height + 0.1*height*math.cos(0.5*math.pi*0.2), 0.5*width - 0.1*height*math.sin(0.5*math.pi*0.2))]
        elif raffle == 6:
            zp = [complex(1, 0.5*width), complex(0.5*height, 0.9*width), complex(height-1, 0.5*width)]
            wa = [complex(height-1, 0.5*width), complex(0.5*height, 0.1*width), complex(1, 0.5*width)]
        elif raffle == 7:
            zp = [complex(0.1*height, 0.5*width), complex(0.5*height, 0.8*width), complex(0.9*height, 0.5*width)]
            wa = [complex(height-1, 0.5*width), complex(0.5*height, 0.1*width), complex(1, 0.5*width)]

        a = np.linalg.det([[zp[0]*wa[0], wa[0], 1],
                            [zp[1]*wa[1], wa[1], 1],
                            [zp[2]*wa[2], wa[2], 1]])
        b = np.linalg.det([[zp[0]*wa[0], zp[0], wa[0]],
                            [zp[1]*wa[1], zp[1], wa[1]],
                            [zp[2]*wa[2], zp[2], wa[2]]])
        c = np.linalg.det([[zp[0], wa[0], 1],
                            [zp[1], wa[1], 1],
                            [zp[2], wa[2], 1]])
        d = np.linalg.det([[zp[0]*wa[0], zp[0], 1],
                            [zp[1]*wa[1], zp[1], 1],
                            [zp[2]*wa[2], zp[2], 1]])

        return a, b, c, d
