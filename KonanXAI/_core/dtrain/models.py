import dtrain as dt
import dtrain.nn as nn

__all__ = ["vgg19", "resnet50"]
pretrained_weights = {
    'vgg16': "xai_mnist_vgg19_for_gradcam.dts",
    'vgg19': "vgg_mnist_full_epoch10.dts",
    'resnet50': "cifar10_resnet50_e1.dts",
}

"""
    VGG
"""
def _vgg_layer(layers):
    seq = []
    layer_id = 0
    for i, (iteration, ksize, ychn, maxpool) in enumerate(layers):
        for _ in range(iteration):
            seq.append(nn.Conv2d(ksize=ksize, ychn=ychn, name=f"conv_{layer_id}"))
            seq.append(nn.ReLU())
            layer_id += 1
        if maxpool:
            seq.append(nn.Max(stride=2))
    return nn.Sequential(*seq)

def _create_vgg(layers, out_width=1000):
    nn.register_macro("features", _vgg_layer(layers))
    nn.register_macro("classifier", nn.Sequential(
        nn.Linear(out_width=4096),
        nn.ReLU(),
        nn.Linear(out_width=4096),
        nn.ReLU(),
        nn.Linear(out_width=out_width)
    ))
    model = nn.Sequential(
        nn.Macro("features", name="features"),
        nn.AdaptiveAvg(sshape=(7, 7), name="gap"),
        nn.Flatten(name="flatten"),
        nn.Macro("classifier", name="classifier")
    )
    return model

def vgg16(session, pretrained=False, out_width=1000):
    if pretrained:
        return session.load_module(pretrained_weights["vgg16"], "cuda" if dt.cuda.device_count() > 0 else "cpu")
    layers = [
        (2, 3, 64, True),
        (2, 3, 128, True),
        (3, 3, 256, True),
        (3, 3, 512, True),
        (3, 3, 512, True),
    ]
    return _create_vgg(layers, out_width)

def vgg19(session, pretrained=False, out_width=1000, weights_path='None'):
    if pretrained:
        return session.load_module(pretrained_weights["vgg19"] if weights_path == 'None' else weights_path, "cuda" if dt.cuda.device_count() > 0 else "cpu")
    layers = [
        (2, 3, 64, True),
        (2, 3, 128, True),
        (4, 3, 256, True),
        (4, 3, 512, True),
        (4, 3, 512, True),
    ]
    return _create_vgg(layers, out_width)

"""
    ResNet
"""
def _create_resnet(model_name, layers, out_width=1000, basic=False):
    nn.register_macro("ResidualBlock", _bottleneck() if not basic else _basic_block())
    nn.register_macro("Layer1", nn.Sequential(
        nn.Conv2d(ksize=7, stride=2, chn=64, actfunc="none"),
        nn.BatchNorm(),
        nn.ReLU(),
        nn.Max(stride=2),
        {"name": "layer1"}
    ))
    for i, layer in enumerate(layers):
        nn.register_macro(f"Layer{i+2}", _resnet_layer(i+2, *layer))
    seq = []
    for i in range(len(layers) + 1):
        seq.append(nn.Macro(f"Layer{i+1}"))
    nn.register_macro(model_name, nn.Sequential(*seq))
    model = nn.Sequential(
        nn.Macro(model_name, name="features"),
        nn.GlobalAvg(name="gap"),
        nn.Linear(width=out_width)
    )
    return model

def _bottleneck():
    bottleneck = nn.Residual(
        nn.Conv2d(ksize=1, stride=1, chn="#", actfunc="none"),
        nn.BatchNorm(),
        nn.ReLU(),
        nn.Conv2d(ksize=3, stride=1, chn="#"),
        nn.BatchNorm(),
        nn.ReLU(),
        nn.Conv2d(ksize=1, stride=1, chn="#chn*4")
    )
    return bottleneck

def _basic_block():
    basic = nn.Residual(
        nn.Conv2d(ksize=3, stride=1, chn="#", actfunc="none"),
        nn.BatchNorm(),
        nn.ReLU(),
        nn.Conv2d(ksize=3, stride=1, chn="#"),
        # nn.BatchNorm(),
        # nn.ReLU(),
    )
    return basic

def _resnet_layer(idx, chn, repeat, max_layer=True):
    seq = []
    seq.append(nn.Macro("ResidualBlock", chn=chn, repeat=repeat))
    if max_layer:
        seq.append(nn.Max(stride=2))
    seq.append({"name": f"layer{idx}"})
    return nn.Sequential(*seq)

def resnet50(session, pretrained=False, out_width=1000, weights_path='None'):
    if pretrained:
        return session.load_module(pretrained_weights["resnet50"] if weights_path == 'None' else weights_path, "cuda" if dt.cuda.device_count() > 0 else "cpu")
    layers = [
        (64, 3, True),
        (128, 4, True),
        (256, 6, True),
        (512, 3, False),
    ]
    model = _create_resnet("Resnet50", layers, out_width=out_width)
    return model

