import torch
import torch.nn as nn
import torch.nn.functional as F
from torchattacks.attack import Attack
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.metrics import mean_squared_error, structural_similarity
import numpy as np
import json
import os
import sys
import time
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from utils import imshow, image_folder_custom_label
import matplotlib.pyplot as plt

# True  False
show = False
original_attack = False

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
# Using normalization for Inception v3.
# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],                     
#                          std=[0.229, 0.224, 0.225])
    
# However, DO NOT USE normalization transforms in this section.
# torchattacks only supports images with a range between 0 and 1.
# Thus, please refer to the model construction section.
    
])

print("dataset:   imagenet-mini_val")
# print("dataset:   imagenet-mini_train")
# print("dataset:   custom mini dataset")
normal_data = image_folder_custom_label(root='./data/imagenet2', transform=transform, idx2label=idx2label)
normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=1, shuffle=False)

test_set = torchvision.datasets.ImageNet( root='./data/imagenet', split= 'val', download=False, transform=transform )
normal_loader =  torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

# Adding a normalization layer for Inception v3.
# We can't use torch.transforms because it supports only non-batch images.
norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


print("original_attack",end=": ")
print(original_attack)

print("network model",end=": ")
model = nn.Sequential(
    norm_layer,
    # models.inception_v3(pretrained=True)
    # models.alexnet(pretrained=True)
    models.resnet50(pretrained=True)
)
print(model)
model = model.to(device).eval()

class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
        
    """
    def __init__(self, model, eps=0.007):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        cost = self._targeted*loss(outputs, labels).to(self.device)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]


        #########
        purturb = grad.sign()
        # filterd = torch.clamp(filterd, min=0.02, max=0.3)
        
        if original_attack == False:
            # adv_images = images_ + purturb_*self.eps
            # eps_filterd_2d = (filterd/filterd.mean())
            # eps_filterd_2d = filterd
            # purturb_ = np.uint8(purturb.cpu().data.squeeze(0).permute(1, 2, 0).numpy()*255)*10
            # purturb_[:,:,0] =  purturb_[:,:,0] * (eps_filterd_2d)*200
            # purturb_ = torch.clamp(transforms.ToTensor()(purturb_), min=0, max=1).detach()
            adv_images = images + filterd*purturb*self.eps
        else:
            adv_images = images + purturb*self.eps


        # adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
class RFGSM(Attack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]
    """
    def __init__(self, model, eps=16/255, alpha=8/255, steps=1):
        super(RFGSM, self).__init__("RFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach() + self.alpha*torch.randn_like(images).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = self._targeted*loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            purturb = (self.eps-self.alpha)*grad.sign()
            if original_attack == False:
                adv_images = adv_images.detach() +  filterd * purturb
            else:
                adv_images = adv_images.detach() + purturb

            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        return adv_images 
class FFGSM(Attack):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]
    
    """
    def __init__(self, model, eps=8/255, alpha=10/255):
        super(FFGSM, self).__init__("FFGSM", model)
        self.eps = eps
        self.alpha = alpha

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        adv_images = adv_images + torch.randn_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.model(adv_images)
        cost = self._targeted*loss(outputs, labels).to(self.device)

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        purturb = self.alpha*grad.sign()

        if original_attack == False:
            adv_images = adv_images.detach() + filterd * purturb
        else:
            adv_images = adv_images.detach() + purturb

        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()


        return adv_images
class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        
        # eps_filterd_2d = (filterd/filterd.mean())*self.eps #이러면 합이 엡실론

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted*loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            
            purturb = self.alpha*grad.sign()
            if original_attack == False:
                adv_images = adv_images.detach() +  filterd * purturb
            else:
                adv_images = adv_images.detach() + purturb

            delta = torch.clamp(adv_images.to(self.device) - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
class TPGD(Attack):
    r"""
    PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]     
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=7):
        super(TPGD, self).__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._attack_mode = 'only_original'

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        
        adv_images = images.clone().detach() + 0.001*torch.randn_like(images).to(self.device).detach()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction='sum')

        for i in range(self.steps):
            adv_images.requires_grad = True
            logit_ori = self.model(images)
            logit_adv = self.model(adv_images)

            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1)).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images.to(self.device) - images, min=-self.eps, max=self.eps)
            if original_attack == False:
                adv_images = torch.clamp(images + filterd * delta, min=0, max=1).detach()
            else:
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            

        return adv_images
class APGD(Attack):
    r"""
    Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
    [https://arxiv.org/abs/1907.00895]
    
    Distance Measure : Linf

    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, sampling=10):
        super(APGD, self).__init__("APGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.sampling = sampling

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for i in range(self.steps):

            grad = torch.zeros_like(images)
            images.requires_grad = True

            for j in range(self.sampling):

                outputs = self.model(images)
                cost = self._targeted*loss(outputs, labels).to(self.device)

                grad += torch.autograd.grad(cost, images,
                                            retain_graph=False,
                                            create_graph=False)[0]

            # grad.sign() is used instead of (grad/sampling).sign()
            adv_images = images + self.alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)

            if original_attack == False:
                images = torch.clamp(ori_images + filterd *eta, min=0, max=1).detach()
            else:
                images = torch.clamp(ori_images + eta, min=0, max=1).detach()

        adv_images = images

        return adv_images
class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

        
    """
    def __init__(self, model, steps=3):
        super(DeepFool, self).__init__("DeepFool", model)
        self.steps = steps
        self._attack_mode = 'only_original'

    def forward(self, images, labels, filterd):
        r"""
        Overridden.
        """
        images = images.to(self.device)

        for b in range(images.shape[0]):

            image = images[b:b+1, :, :, :]

            image.requires_grad = True
            output = self.model(image)[0]

            _, pre_0 = torch.max(output, 0)
            f_0 = output[pre_0]
            grad_f_0 = torch.autograd.grad(f_0, image,
                                           retain_graph=False,
                                           create_graph=False)[0]
            num_classes = len(output)

            for i in range(self.steps):
                image.requires_grad = True
                output = self.model(image)[0]
                _, pre = torch.max(output, 0)

                if pre != pre_0:
                    image = torch.clamp(image, min=0, max=1).detach()
                    break

                r = None
                min_value = None

                for k in range(num_classes):
                    if k == pre_0:
                        continue

                    f_k = output[k]
                    grad_f_k = torch.autograd.grad(f_k, image,
                                                   retain_graph=True,
                                                   create_graph=True)[0]

                    f_prime = f_k - f_0
                    grad_f_prime = grad_f_k - grad_f_0
                    value = torch.abs(f_prime)/torch.norm(grad_f_prime)

                    if r is None:
                        r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
                        min_value = value
                    else:
                        if min_value > value:
                            r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
                            min_value = value

                if original_attack == False:
                    image = torch.clamp(image + filterd * r, min=0, max=1).detach()
                else:
                    image = torch.clamp(image + r, min=0, max=1).detach()
                

            images[b:b+1, :, :, :] = image

        adv_images = images

        return adv_images

attacks = [
           FGSM(model, eps=4/255),
           FFGSM(model, eps=4/255, alpha=12/255),
           RFGSM(model, eps=8/255, alpha=4/255, steps=1),
           PGD(model, eps=4/255, alpha=2/255, steps=7),
           APGD(model, eps=4/255, alpha=2/255, steps=7),
           TPGD(model, eps=8/255, alpha=2/255, steps=7),
           DeepFool(model, steps=3),
           
           #torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=1),
           #torchattacks.FFGSM(model, eps=8/255, alpha=12/255),
           #torchattacks.APGD(model, eps=8/255, alpha=2/255, steps=7),
           #torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7),

          ]


def filter(im):
    # read image
    imsum = im.sum(axis=2)
    img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    img_canny = cv2.Canny(img_gray, 50, 150)
    img_canny_f = img_as_float(img_canny)
        
    N=5
    S=img_canny.shape
    E=np.array(img_canny_f)
    for row in range(S[0]):
            for col in range(S[1]):
                    Lx=np.max([0,col-N])
                    Ux=np.min([S[1],col+N])
                    Ly=np.max([0,row-N])
                    Uy=np.min([S[0],row+N])
                    region=img_canny_f[Ly:Uy,Lx:Ux].mean()
                    E[row,col]=region
    E = E + 0.02
    if show == True:
        plt.imshow(E, cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()  
    
        # print(img_canny_f.mean())
    E = E / E.mean()
                
    return E

print("Adversarial Image & Predicted Label")

for attack in attacks :
    
    print("-"*70)
    print(attack)
    
    correct = 0
    total = 0
    stacked_img =  np.array([[0]*3])
    for images_, labels in normal_loader:
        original = np.uint8(images_.squeeze(0).permute(1, 2, 0).numpy()*255)
        start = time.time()

        if original_attack == False:
            filterd = filter(original)
            
            stacked_img = np.stack((filterd,)*3,-1)
            
        adv_images = attack(images_, labels, transforms.ToTensor()(stacked_img).to(device, dtype=torch.float))
        
        # print(structural_similarity(original,np.uint8(adv_images.clone().cpu().squeeze(0).permute(1, 2, 0).numpy()*255), full=True,multichannel=True))
        
        if show == True:
            # imshow(torchvision.utils.make_grid(transforms.ToTensor()(adv_images).permute(0, 1, 2).cpu().data),  [normal_data.classes[i] for i in pre])
            
            imshow(torchvision.utils.make_grid(adv_images.cpu().data),'filterd')
            #plt.imshow(original)
            #plt.show()

        # img = np.array([originaㅋl])
        labels = labels.to(device)
        outputs = model(adv_images.to(device))

        _, pre = torch.max(outputs.data, 1)
        total += 1
        correct += (pre == labels).sum()
        """
        if (pre == labels):
            print('O',end=" ")
        else:
            print('X',end=" ")
        """
        # imshow(torchvision.utils.make_grid(transforms.ToTensor()(original).permute(0, 1, 2).cpu().data),  [normal_data.classes[i] for i in pre])

        # imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])
        # imshow(torchvision.utils.make_grid(noise_.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])

    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))

 