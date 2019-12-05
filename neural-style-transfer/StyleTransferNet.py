import torch
import torch.nn as nn
import torchvision as tv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Layer Name to index map to easily reference features
layer_to_index = dict(r11=1, r12=3, r21=6, r22=8, r31=11, r32=13, r33=15, r34=17, r41=20, r42=22, r43=24, r44=26,
                      r51=29, r52=31, r53=33, r54=35)


## Main Neural Net
class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()

        self.vgg = tv.models.vgg19(pretrained=True)

        for param in self.vgg.parameters():
            param.requires_grad = False

    ## Runs the forward method on pretrained vgg net and returns output features for the passed layer names
    def forward(self, x, out_layers):
        output = {}
        output[-1] = x
        num_of_layers = len(self.vgg.features)
        layers = self.vgg.features

        for i in range(num_of_layers):
            output[i] = layers[i](output[i - 1])

        out_layer_indices = [layer_to_index[layer] for layer in out_layers]

        return [output[key] for key in out_layer_indices]


# Gram matrix and Loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out

# Initialize Neural Net and Loss Functions and targets
def init(style_image, content_image, style_layers, content_layers):

    net = StyleTransferNet().to(device)

    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in net(style_image, style_layers)]
    content_targets = [A.detach() for A in net(content_image, content_layers)]
    targets = style_targets + content_targets

    return net, loss_layers, loss_fns, targets