from datasampler import CIFAR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from model import VIT
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter

def train(dataloader, device, num_epochs):
    Losses = []
  
    lr = 0.001
    beta1 = 0.5
    iter = 0
    z_dim, channels_img, f_g = 100, 3, 32
    img_shape = 32,32,3
    f_d = 32
    patch_dim = 4
    embed_dim = 32
    # writer = SummaryWriter(f"logs/loss")
    algo = VIT(img_shape, patch_dim, embed_dim, img_shape[-1] , 6, 2, 10, 0.1).to(device=device)
    opt = torch.optim.AdamW(algo.parameters(), lr=lr, betas=[beta1, 0.999])
    criterion = nn.CrossEntropyLoss()
    
    
    for epoch in range(num_epochs):
        epoch_losses = []
        tot = 0
        correct = 0
        for i, (img, labels) in enumerate(dataloader):
            opt.zero_grad()
            output = algo(img)
            loss = criterion(output, labels)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
            _, predicted_labels = torch.max(output, 1)
            _, true_labels = torch.max(labels, 1)
            tot += len(predicted_labels)
            correct += (predicted_labels == true_labels).float().sum()
            accuracy = correct/tot
             
            if i%50 ==0:
                print(f'Epoch:{epoch}  {i}/{len(dataloader)}  Loss:{np.array(epoch_losses).mean()} Accuracy:{accuracy}')
        # writer.add_scalar('loss',np.array(epoch_losses).mean(),epoch)
       
            
    # return Gen_Losses, Dis_Losses
    

def main():
    path = '/Volumes/E/PapersWithCode/GAN/Data/cifar-10-batches-py'
    device = torch.device("mps")
    data = CIFAR(path)
    images, labels = data.get_train_data()
    images_tensor = torch.tensor(images,dtype=torch.float32).to(device)
    labels_tensor = F.one_hot(torch.tensor(labels).to(torch.int64)).to(torch.float32).to(device)
    
    
    my_dataset = TensorDataset(images_tensor,labels_tensor) 
    my_dataloader = DataLoader(my_dataset,shuffle=True, batch_size=128) 
    train(my_dataloader,device,5)
    
    
if __name__ == "__main__":
    main()