"""A stand-in for lerobot's SceneEncoder in the cursor tests (imported by bare name: `from _view_stub import ViewStub`)."""

import torch

from lerobot.policies.steering import SceneEncoder


class ViewStub(SceneEncoder):
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


class TokenStub(SceneEncoder):
    """Stands in for SceneEncoder's `image_tokens` / `prefix_tokens`: float16 [B, 256, 2048], each token a fixed
    nonlinear function of its 16 x 16 patch grid cell's mean colour (a local generator's weights, so torch's global
    streams are never touched). Counts the pictures it encoded (`images`)."""

    def __init__(self, *a, **k):
        self.images = 0
        self.weight = torch.randn(3, 2048, generator=torch.Generator().manual_seed(0))

    def image_tokens(self, images):
        self.images += len(images)
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)  # [B, 256, 3]
        return torch.tanh(3 * pooled @ self.weight).to(torch.float16).numpy()

    def prefix_tokens(self, images):
        return -self.image_tokens(images)

    def close(self):
        pass
