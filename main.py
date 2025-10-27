from asyncio import FastChildWatcher
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F

import pickle as pk
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix,f1_score
from argparse import ArgumentParser
from models import *

from doa import *
import random

def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
import wandb
def main(args):
    setseed(args.seed)
    if args.wandb:
        wandb.init(project=args.data_name, name=f"{args.data_name}_{args.wandb_info}__train_val_{args.train_val}_optim_shce_{args.optim_sche}_momentum_{args.momentum}_gamma_{args.gamma}_step_size_{args.step_size}_seed_{args.seed}_lr_{args.lr}_batch_{args.batch_size}_rate_{args.rates}_coder_number_{args.coder_number}_block_number_{args.block_number}_dim_{args.dim}_fksk_{args.fk}_{args.sk}.pt")
    if args.save_info:
        r_matrix = np.load(f"/datasets/{args.data_name}/r_matrix.npy")
        q_matrix = pd.read_csv(f"/datasets/{args.data_name}/q_matrix.csv",header=None).to_numpy()[1:,]
        q_diff = pk.load(open(f"/data/{args.data_name}/q_diff.pkl",'rb'))
        k_pos = pk.load(open(f"/data/{args.data_name}/k_pos.pkl",'rb'))
        print("finish_read r & q")
    data_name = args.data_name 
    diff_dim = args.diff_dim

    window_list = []
    for i in range(args.window_size+1):
        if i!=0:
            window_list.append(-1*i)
            window_list.append(i)
        else:
            window_list.append(0)
    
    class Generator(nn.Module):
        def __init__(self,dim, coder_number, rates, block_num,bm,kernel=(2,4)):
            super(Generator, self).__init__()
            self.dim=dim
            self.coder_number = coder_number
            self.rates = [pow(2,i) for i in range(rates)] # 0,1,2,3->1,2,4,8
            self.block_num = block_num
            self.bm = bm
            dropout_prob=0.2
            self.encoder = nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=kernel, stride=(2,1), padding=1, bias=False),  
                nn.BatchNorm2d(dim, self.bm),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(dim, dim, kernel_size=kernel, stride=(2,1), padding=1, bias=False), 
                nn.BatchNorm2d(dim,self.bm),
                nn.LeakyReLU(0.2, inplace=True),
            )

            for i in range(self.coder_number):  
                self.encoder.add_module(f'encoder_{i}', nn.Sequential(
                    nn.Conv2d(dim*(2**i), dim * (2 ** (i+1)), kernel_size=kernel, stride=(2, 1), padding=1, bias=False),  
                    nn.BatchNorm2d(dim * (2 ** (i+1)), self.bm),
                    nn.LeakyReLU(0.2, inplace=True),
                ))
            
            self.middle = nn.Sequential(*[MultiBlock(dim*(2**self.coder_number), self.rates,0.1) for _ in range(self.block_num)])
            self.decoder = nn.Sequential()
            for i in range(self.coder_number-1,-1,-1):  
                self.decoder.add_module(f'decoder_{i}', nn.Sequential(
                    nn.ConvTranspose2d(dim*(2**(i+1)), dim * (2 ** i), kernel_size=kernel, stride=(2,1), padding=1, output_padding=(1, 0), bias=False),  
                    nn.BatchNorm2d(dim * (2 ** i), self.bm),
                    nn.LeakyReLU(0.2, inplace=True),
                ))
            self.decoder.add_module(f'decoder_end', nn.Sequential(

                nn.ConvTranspose2d(dim, dim, kernel_size=kernel, stride=(2,1), padding=1, output_padding=(1, 0), bias=False),  
                nn.BatchNorm2d(dim, self.bm),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(dim, 1, kernel_size=kernel, stride=(2,1), padding=1, output_padding=(1, 0), bias=False), 
                nn.Sigmoid(),
            )
            )
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
    
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
              
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
           
                    torch.nn.init.constant_(m.weight, 1.0)
                    torch.nn.init.constant_(m.bias, 0.0)    
        def forward(self, x):
            x = self.encoder(x)
            x = self.middle(x)
            x = self.decoder(x)

            return x
        def apply_row_decreasing(self, x):

            batch_size, channels, height, width = x.size()

            row_gradient = torch.linspace(1, 0, width).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x.device)
            row_gradient = row_gradient.expand(batch_size, channels, height, width)
            
            x = x * row_gradient
            return x
    valid_student = pk.load(open(f"/data/{data_name}/{diff_dim}_valid_student_info_train_val_{args.train_val}.pkl",'rb'))
    test_student = pk.load(open(f"/data/{data_name}/{diff_dim}_test_student_info_train_val_{args.train_val}.pkl",'rb'))

    train_val_student = pk.load(open(f"/data/{data_name}/{diff_dim}_train_val_student_info_train_val_{args.train_val}.pkl",'rb'))
    bce_loss = nn.BCELoss()

    def compute_D(images,students):
        pre=[]
        tar=[]
        for chart,u in zip(images,students):
            chart=chart.squeeze(0)
            u=u.item()

            if int(u) not in train_val_student:
                continue
            infos = train_val_student[int(u)]

            for i,info in enumerate(infos):
                pre.append(chart[info[0],info[1]])
                tar.append(info[2])

        pre = [x.unsqueeze(0) for x in pre]
        pre=torch.cat(pre)
        tar=torch.tensor(tar,dtype=torch.float32).cuda()

        diff1 = images[:, :, :, 1:] - images[:, :, :, :-1] 
        monotonicity_loss = torch.relu(diff1).mean() 
        return bce_loss(pre,tar) +  args.loss_w*monotonicity_loss


    def valid_compute_D(images,students):
        pre=[]
        tar=[]
        for chart,u in zip(images,students):

            chart=chart.squeeze(0).detach().cpu().numpy()
            u=u.item()
            if u not in valid_student:
                continue
            infos=valid_student[u] 
            for info in infos:
                pre_avg=[]
                for i in  window_list:
                    if info[1]+i>=0 and info[1]+i<=args.diff_dim:
                        pre_avg.append(chart[info[0],info[1]+i])
                pre.append(sum(pre_avg)/len(pre_avg))
                tar.append(info[2])
        return pre,tar
        
    def test_compute_D(images,students):
        pre=[]
        tar=[]
        for chart,u in zip(images,students):
            chart=chart.squeeze(0).detach().cpu().numpy()
            u=u.item()
            if u not in test_student:
                continue
            infos=test_student[u] 
            for info in infos:
                pre_avg=[]
                for i in  window_list:
                    if info[1]+i>=0 and info[1]+i<=args.diff_dim:
                        pre_avg.append(chart[info[0],info[1]+i])
                pre.append(sum(pre_avg)/len(pre_avg))
                tar.append(info[2])

        return pre,tar

    def loss_function(input_images, generated_images,students): 
        loss_bce = compute_D(generated_images,students)
        return loss_bce

    class CustomDataset(Dataset):
        def __init__(self, image_file, target_file,percent=0.1):
            self.images = torch.load(image_file).float()
            self.targets = torch.load(target_file)
            total_data = len(self.images)
            print("student number:", total_data)
            select_count = int(total_data * percent)
            indices = torch.randperm(total_data)[:select_count]
            self.images = self.images[indices]
            self.targets = self.targets[indices]
        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            idx = idx
            image = self.images[idx].unsqueeze(0) 
            target = self.targets[idx]
            return image, target

    dataset = CustomDataset(f'/data/{data_name}/{diff_dim}_images_data_train_val_{args.train_val}.pt', f'/data/{data_name}/{diff_dim}_targets_data_train_val_{args.train_val}.pt',args.percent)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataloader_val = dataloader
    all_test_auc,all_test_acc,all_test_cm,all_test_f1,all_test_rmse=0,0,0,0,0
    kernels=(2,4)
    best_students_raw_images=dict()
    all_students_gene_images=dict()
    best_student_info=dict()
    
    for fk in [args.fk]:
        for sk in [args.sk]:
            try:
                print("===========(",fk,"  ",sk,")================")
                G = Generator(args.dim,args.coder_number, args.rates, args.block_number,args.bm, kernel=(fk,sk)).cuda()
                if args.optim_sche==1:
                    optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1,args.b2))
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
                elif args.optim_sche ==2:
                    optimizer = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.Tm)
                elif args.optim_sche == 3:
                    optimizer = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
                elif args.optim_sche == 4:
                    # optimizer = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
                    optimizer = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum)

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
                elif args.optim_sche == 5:
                    # optimizer = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
                    optimizer = torch.optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum)

                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
                elif args.optim_sche == 6:
                    optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
                    # optimizer = optim.SGD(G.parameters(), lr=0.02, momentum=0.9)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
                elif args.optim_sche == 7:
                    optimizer = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
                elif args.optim_sche == 8:
                    optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1,args.b2), weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
                elif args.optim_sche == 9:
                    optimizer = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)

                num_epochs = args.epoch
                valid_auc_list,valid_acc_list,valid_cm_list=0,0,[]
                test_auc_list,test_acc_list,test_cm_list=0,0,[]
                best_students_gene_images=dict()
                for epoch in range(num_epochs):
                    epoch_loss = 0 
                    G.train()
                    for i, (input_image, students) in enumerate(dataloader):  
                        input_image = input_image.cuda()
                        generated_image = G(input_image)
                     
                        loss_bce = loss_function(input_image,generated_image,students) 
                        loss = loss_bce
                       
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=4.0)
                        optimizer.step()
                        epoch_loss+=loss.item()
                    
                    # scheduler.step(epoch_loss)
                    
                    scheduler.step()

                    avg_epoch_loss = epoch_loss / len(dataloader)  
                    # print(f'Epoch [{epoch+1}/{num_epochs}] Total Loss: {avg_epoch_loss}')
                    G.eval()
                    students_gene_images=dict()
                    # students_raw_images=dict()
                    # student_info=dict()
                    with torch.no_grad():
                        test_pre_all,test_tar_all=[],[]
                        valid_pre_all,valid_tar_all=[],[]
                        for i, (input_image, students) in enumerate(dataloader_val): 
                            input_image = input_image.cuda()
                            generated_image = G(input_image) 
                            
                            pre,tar=test_compute_D(generated_image,students)
                            test_pre_all.extend(pre)
                            test_tar_all.extend(tar)

                            pre,tar=valid_compute_D(generated_image,students)
                            valid_pre_all.extend(pre)
                            valid_tar_all.extend(tar)

                            for i,s in enumerate(students):
                                students_gene_images[int(s)]=generated_image[i].detach().cpu()
                                # students_raw_images[int(s)]=input_image[i].detach().cpu()

                        binary_pred =  np.array(valid_pre_all) >= 0.5
                        valid_auc,valid_acc,valid_f1_score,valid_rmse,valid_cm=roc_auc_score(valid_tar_all, valid_pre_all), accuracy_score(valid_tar_all, binary_pred ), f1_score(valid_tar_all, binary_pred ), np.sqrt(np.mean((np.array(valid_pre_all) - np.array(valid_tar_all)) ** 2)), confusion_matrix(valid_tar_all,binary_pred)
                        binary_pred =  np.array(test_pre_all) >= 0.5
                        test_auc,test_acc,test_f1_score,test_rmse,test_cm=roc_auc_score(test_tar_all, test_pre_all), accuracy_score(test_tar_all, binary_pred ), f1_score(test_tar_all, binary_pred), np.sqrt(np.mean((np.array(test_pre_all) - np.array(test_tar_all)) ** 2)), confusion_matrix(test_tar_all, binary_pred )

                        if valid_acc + valid_auc > valid_acc_list+valid_auc_list:
                            valid_auc_list=valid_auc
                            valid_acc_list=valid_acc
                            valid_cm_list=valid_cm
                            test_auc_list=test_auc
                            test_acc_list=test_acc
                            test_cm_list=test_cm
                            test_f1_list=test_f1_score
                            test_rmse_list=test_rmse
                        
                            best_students_gene_images = students_gene_images

                        if args.wandb:
                                wandb.log({
                                    "epoch": epoch,
                                    "loss":  avg_epoch_loss,                                 
                                    "valid_auc": valid_auc,
                                    "valid_acc": valid_acc,
                                    "valid_f1": valid_f1_score,
                                    "valid_rmse": valid_rmse,
                                    "test_auc": test_auc,
                                    "test_acc": test_acc,
                                    "test_f1:": test_f1_score,
                                    "test_rmse": test_rmse,
                                    "lr": scheduler.get_last_lr()[0],
                                })
                if test_acc_list>all_test_acc:
                    all_test_auc=test_auc_list
                    all_test_acc=test_acc_list
                    all_test_cm=test_cm_list
                    all_test_f1=test_f1_list
                    all_test_rmse=test_rmse_list
                    kernels=(fk,sk)
                    print("best")
                    print(kernels,all_test_auc,all_test_acc,all_test_cm)
                    all_students_gene_images=best_students_gene_images # check this 
                    print("save_info")
            except:
                print(f"error:{fk},{sk}")
                continue
    print(kernels,all_test_auc,all_test_acc,all_test_f1,all_test_rmse,all_test_cm)
    if args.save_info:
        if args.data_name == "ednet":
            student_number = 4990
        elif args.data_name == "a2012":
            student_number = 27485
        elif args.data_name in ["a0405"]:
            student_number = 574
        elif args.data_name in ["a2017"]:
            student_number = 1708
        else:
            student_number = 18066
        image_list=[]
        for x in range(1,student_number+1):
            if x not in all_students_gene_images:
                all_students_gene_images[x]=all_students_gene_images[x-1]
            image_list.append(all_students_gene_images[x])
        students_images_tensor = torch.stack(image_list, dim=0)  
        torch.save(students_images_tensor, f"student_info/{args.data_name}/{args.wandb_info}_{args.seed}_train_val_{args.train_val}_bs_{args.batch_size}_dim_{args.dim}_{args.fk}_{args.sk}_students_images_tensor_0210.pt")  
        
        doa_info = DOA( students_images_tensor.numpy(),k_pos,q_diff, q_matrix, r_matrix, data_name)
        print(doa_info)
if __name__ == "__main__":
    parser = ArgumentParser(description="Run the NCDM model with specified dataset.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--epoch", type=int, required=False,default=30, help="epoch")

    parser.add_argument("--diff_dim", type=int, required=False,default=20, help="diff dim")
    parser.add_argument("--percent", type=float, required=False,default=1.0, help="percent")
    parser.add_argument("--lr", type=float, required=False,default=0.02, help="lr")

    parser.add_argument("--dim", type=int, required=False,default=64, help="dim")
    parser.add_argument("--coder_number", type=int, required=True, help="coder_number")
    parser.add_argument("--rates", type=int, required=True, help="rates")
    parser.add_argument("--block_number", type=int, required=True, help="block_number")
    parser.add_argument("--random_number", type=int, required=False,default=10, help="random_number")
    parser.add_argument("--loss_w", type=float, required=False,default=10, help="loss_w")
    parser.add_argument("--cl_w", type=float, required=False,default=0.01, help="loss_w")
    parser.add_argument("--save_info", type=int, required=False,default=1, help="loss_w")
    parser.add_argument("--wandb_info", type=str, required=False,default="none", help="window size")
    parser.add_argument("--wandb", type=int, required=False,default=1, help="window size")
    parser.add_argument("--optim_sche", type=int, required=False,default=1, help="optim sche")
    parser.add_argument("--T0", type=int, required=False,default=10, help="optim sche")
    parser.add_argument("--Tm", type=int, required=False,default=1, help="optim sche")
    parser.add_argument("--b1", type=float, required=False,default=0.5, help="beta1")
    parser.add_argument("--b2", type=float, required=False,default=0.999, help="beta1")
    parser.add_argument("--momentum", type=float, required=False,default=0.9, help="beta1")
    parser.add_argument("--gamma", type=float, required=False,default=0.5, help="beta1")

    parser.add_argument("--train_val", type=float, required=False,default=0.6, help="train_val")

    parser.add_argument("--step_size", type=int, required=False,default=10, help="step_size")

    parser.add_argument("--window_size", type=int, required=False,default=1, help="window_size")
    parser.add_argument("--batch_size", type=int, required=False,default=256, help="batch_size")
    parser.add_argument("--fk", type=int, required=False,default=4, help="fk")
    parser.add_argument("--sk", type=int, required=False,default=4, help="sk")
    parser.add_argument("--seed", type=int, required=False,default=3702, help="seed")

    parser.add_argument("--bm", type=float, required=False,default=0.01, help="seed")

    parser.add_argument("--train_ratio", type=float, required=False,default=0.01, help="seed")
    
    args = parser.parse_args()
    print(args)
    main(args)
    print("=======finish======")
