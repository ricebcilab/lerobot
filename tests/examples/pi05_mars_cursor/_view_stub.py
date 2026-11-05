"""A stand-in for lerobot's SceneEncoder in the cursor tests (imported by bare name: `from _view_stub import ViewStub`)."""

import torch


class ViewStub:
    """Stands in for SceneEncoder: token j's first three dims are patch j's mean colour / 255. Counts the stubs made
    (`made`, on the class) and the pictures each one encoded (`images`); every `encode` draws once from torch's global
    generator, so a caller that must leave torch's streams alone is tested for it."""

    made = 0

    def __init__(self, *a, **k):
        ViewStub.made += 1
        self.images = 0

    def encode(self, images, batch=256):
        self.images += len(images)
        # one draw from torch's generator, as the real encoder's first load draws its initialisation
        torch.rand(1)
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return torch.nn.functional.pad(pooled, (0, 2045)).to(torch.float16)

    def close(self):
        pass
