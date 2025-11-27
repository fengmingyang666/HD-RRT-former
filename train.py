import torch
import torch.nn as nn
import torch.optim as optim
from model_6D import TransformerNodeSampler6D_PI_KGAN
from dataset import get_data_loader
from tqdm import tqdm
from torchinfo import summary
import sys
import numpy as np
import wandb

import os
# os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader=None,D=6, epochs=100, lr=0.001, device='cpu', model_save_dir='./ckpt_s3'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_dir, f'best_model_{D}d.pt')
    
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            total_train_loss = 0
            model.train()
            pbar.set_description('Epoch {}/{}'.format(epoch, epochs))
            
            for batch_idx, (node_positions, target_points, env_maps) in tqdm(enumerate(train_loader), total=len(train_loader)):
                node_positions = node_positions.to(device)
                target_points = target_points.to(device)
                env_maps = env_maps.to(device)

                optimizer.zero_grad()
                outputs = model(node_positions, env_map=env_maps)
                loss = mse_loss(outputs, target_points)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            avg_val_loss = None
            if val_loader is not None:
                total_val_loss = 0
                model.eval()
                with torch.no_grad():
                    for node_positions, target_points, env_maps in val_loader:
                        node_positions = node_positions.to(device)
                        target_points = target_points.to(device)
                        env_maps = env_maps.to(device)

                        outputs = model(node_positions, env_map=env_maps)
                        loss = mse_loss(outputs, target_points)
                        total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model, best_model_path)
                    print(f"Best model saved with validation loss: {best_val_loss:.6f} at {best_model_path}")

                    wandb.log({"best_model_path": best_model_path, "best_val_loss": best_val_loss})
            

            if epoch % 10 == 0:
                checkpoint_path = os.path.join(model_save_dir, f'model_{D}d_{epoch}.pt')
                torch.save(model, checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

                wandb.log({f"checkpoint_epoch_{epoch}": checkpoint_path})
            
            log_dict = {"train_loss": avg_train_loss}
            if avg_val_loss is not None:
                log_dict["val_loss"] = avg_val_loss
                pbar.set_postfix(train_loss='{:.3f}'.format(avg_train_loss), val_loss='{:.3f}'.format(avg_val_loss))
            else:
                pbar.set_postfix(train_loss='{:.3f}'.format(avg_train_loss))
            
            wandb.log(log_dict)
            pbar.update(1)

if __name__ == "__main__":
    dataset_path = "data/dataset_6D.npy"
    
    model_save_dir = './ckpt'
    os.makedirs(model_save_dir,exist_ok=True)
    
    D=6
    batch_size = 128
    epochs = 2000
    lr = 1e-4
    validation_split = 0.1  

    wandb.init(
        project=f"RRT_Net_{D}D",
        name="pi_kgan_rrt_connect_fliter",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dataset": dataset_path,
            "validation_split": validation_split,
            "model_save_dir": model_save_dir,
        })

    train_loader, val_loader = get_data_loader(dataset_path, batch_size=batch_size, validation_split=validation_split)
    
    dataiter = iter(train_loader)
    s = next(dataiter)
    print(s[0].size(), s[1].size(), s[2].size())  # (batchsize, length, dim), (batchsize, dim), (batchsize, depth, height, width)
    
    model = TransformerNodeSampler6D_PI_KGAN(input_dim=6, d_model=72, nhead=8, num_encoder_layers=6, depth=50, height=50, width=50)

    summary(model, [(32, 500, D), (32, 50, 50, 50)])

    train_model(model, train_loader, val_loader, D=D, epochs=epochs, lr=lr, device=device, model_save_dir=model_save_dir)
    
    final_model_path = os.path.join(model_save_dir, f'model_{D}d.pt')
    wandb.log({"final_model_path": final_model_path})
    
    wandb.finish()
    torch.save(model, final_model_path)
    print(f"Model saved done at: {final_model_path}")