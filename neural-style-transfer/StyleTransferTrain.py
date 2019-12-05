import torch
from torch import optim
from torchvision import transforms
import StyleTransferNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Post Processing Method to get the output image from tensor
def postProcess(tensor):
    postProcess1 = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                               transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                    std=[1,1,1]),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                               ])
    postProcess2 = transforms.Compose([transforms.ToPILImage()])

    t = postProcess1(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postProcess2(t)
    return img

def train(T, style_layers, content_layers, style_weights, content_weights, num_epochs):
    weights = style_weights + content_weights

    style_image = T[2].to(device)
    content_image = T[3].to(device)
    opt_img = T[4].to(device)

    optimizer = optim.LBFGS([opt_img]);
    net, loss_layers, loss_fns, targets = StyleTransferNet.init(
        style_image, content_image, style_layers, content_layers)

    n_iter = 0
    show_iter = 50
    while n_iter <= num_epochs:
        optimizer.zero_grad()
        out = net(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter += 1
        optimizer.step()

        # print loss
        if n_iter % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter + 1, loss.data[0]))

    out_img = postProcess(opt_img.cpu().squeeze())

    return out_img

