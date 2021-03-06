import os
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as td
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


## Dataset Class used for Style Transfer. Loads a style and content pair, pre-processes them and returns all of them.
## img_size - size of the images used for training.
## img_dir - Image Directory path expecting to have two directories Style/ and Content/ directories. they are expected to have files with same name which would be paired together.
## Init - The way to initialise the opt_image. Possible Values -  "Random", "Content" or "Style"
class StyleTransferDataset(td.Dataset):
    def __init__(self, img_size=512, img_dir = './Images/', init='Random'):
        super(StyleTransferDataset, self).__init__()
        self.img_size = img_size
        style_image_dir = img_dir + 'Style/'
        content_image_dir = img_dir + 'Content/'
        self.style_images_dir = style_image_dir
        self.content_images_dir = content_image_dir
        self.files = sorted(os.listdir(self.style_images_dir))
        self.init = init

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "StyleTransferDataset(image_size={})". \
            format(self.image_size)

    ## Returns
    ## [Style Image, Content Image, PreProcessed Style Image Tensor, PreProcessed Content Image Tensor, OptImage]
    def __getitem__(self, idx):

        style_img_path = os.path.join(self.style_images_dir, self.files[idx])
        content_img_path = os.path.join(self.content_images_dir, self.files[idx])

        ## Referencing Original paper's work
        pre_process = transforms.Compose([transforms.Resize(self.img_size),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                                          transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                               # subtract imagenet mean
                                                               std=[1, 1, 1]),
                                          transforms.Lambda(lambda x: x.mul_(255)),
                                          ])

        imgs = [Image.open(style_img_path), Image.open(content_img_path)]
        imgs_torch = [pre_process(img) for img in imgs]
        imgs_torch = [img.unsqueeze(0).to(device) for img in imgs_torch]

        style_image, content_image = imgs_torch
        
        if self.init is 'Content':
            opt_img = content_image.clone().detach().requires_grad_(True)
        elif self.init is 'Style':
            opt_img = style_image.clone().detach().requires_grad_(True)
        else: ## Default =  Random
            opt_img = torch.randn(content_image.size()).type_as(content_image.data).requires_grad_(True)


        return imgs + [style_image, content_image, opt_img]