import torch
import torch.nn as nn
import numpy as np

def pgd_attack(model, images, device, iteration=20, target=None, 
               clip_min=0, clip_max=1, rand_clip=0.3, l2perturb=0.3):
    
    # Reshape images in 4 dim 
    images = images.reshape(-1,3,299,299)
        
    return torch.stack([pgd_attack_1(model, image, device, iteration, 
                                     target, clip_min, clip_max, rand_clip, 
                                     l2perturb) for image in images])


def pgd_attack_1(model, image, device, iteration, target, 
                 clip_min, clip_max, rand_clip, l2perturb):
    
    image = image.reshape(-1,3,299,299).to(device, dtype=torch.float)
    
    # Random init
    perturb = torch.rand(image.shape).uniform_(-rand_clip, rand_clip)
    perturb = clip_eta(perturb, l2perturb)
    attack_result = image + perturb.to(device, dtype=torch.float)
    attack_result = torch.clamp(attack_result, min=clip_min, max=clip_max)
    attack_result.requires_grad = True
    
    # Switch to eval mode 
    if model.training:
        model.eval()
    label_o = model(image)
    _, predicted = torch.max(label_o.data, 1)
    
    # If target is None, minimize original predict label prob
    if target is None:
        targeted_attack = False
        target = predicted[0]
    else:
        targeted_attack = True
    target = torch.tensor(target, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(iteration):
        
        # FGM 1 step generate example
        if attack_result.grad is not None:
            attack_result.grad.zero_()
            
        label = model(attack_result)
        loss = criterion(label, target.view(-1).to(device, dtype=torch.long))
        loss.backward()
        perturb = optimal_perturb(attack_result.grad.cpu().detach())
        
        # increase target loss (true label) for untargeted attack
        # decrease target loss (target label) for targeted attack
        if targeted_attack:
            attack_result = attack_result - perturb.to(device, dtype=torch.float)
        else:
            attack_result = attack_result + perturb.to(device, dtype=torch.float)
            
        attack_result = torch.clamp(attack_result, min=clip_min, max=clip_max)
        
        # Adjust perturb by l2 norm and momentum.
        perturb = (attack_result - image).cpu()
        perturb = clip_eta(perturb.detach(), l2perturb)
        attack_result = image + perturb.to(device, dtype=torch.float)
        attack_result = torch.clamp(attack_result, min=clip_min, max=clip_max)
        attack_result = attack_result.clone().detach().requires_grad_(True)
    
    return attack_result.detach()


def optimal_perturb(grad, eps=0.3):
    avoid_zero = 1e-10
    # element-wise select max value
    norm = torch.max(torch.tensor(avoid_zero), 
                     torch.sqrt(torch.sum(grad**2, dim=(-1,-2), keepdim=True)))
    optimal_grad = grad/norm
    
    return optimal_grad*eps


def clip_eta(eta, eps):
    avoid_zero = 1e-10
    norm = torch.max(torch.tensor(avoid_zero), 
                     torch.sqrt(torch.sum(eta**2, dim=(-1,-2), keepdim=True)))
    factor = torch.min(torch.tensor(1.), eps/norm)
    eta = eta * factor
    
    return eta