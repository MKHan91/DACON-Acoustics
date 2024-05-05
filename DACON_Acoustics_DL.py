import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import cv2
import sys
import time
import random
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import os.path as osp
import os
import os
# 특정 GPU 번호를 지정하여 CUDA_VISIBLE_DEVICES 환경 변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STFTDataLoader(Dataset):
    def __init__(self, image, mode='train', transform = None):
        self.image = image
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        sample_image = self.image[idx]

        if self.mode == 'train':
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            if self.transform:
                sample_image = self.transform(sample_image)
            return sample_image

        elif self.mode == 'val':
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            sample_image = self.transform(sample_image)
            return sample_image

        elif self.mode == 'test':
            sample_image = np.array(sample_image, dtype=np.float32)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            sample_image = self.transform(sample_image)
            return sample_image
            

# Define the autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 3x128x128 -> 16x64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x64x64 -> 32x32x32
            nn.ReLU()
        )

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 32x32x32 -> 16x64x64
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  # 16x64x64 -> 3x128x128
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train():
    learning_rate = 1e-4
    random_seed = 41
    batch_size = 16
    num_epochs = 100
    set_seeds(random_seed)

            
    raw_stft = np.load('./DATASET/stft_image_dataset.npy')

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(128, 128))

    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(128, 128))

    ])

    # 학습 데이터, 검증 데이터 분리
    kf = KFold(n_splits=5, shuffle=True)

    # dataset = STFTDataLoader(image=raw_stft, mode='train', transform=transform)
    # data_loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    log_folder_name = 'stft_try2'
    log_dir = osp.join(os.getcwd(), 'logs', f'{log_folder_name}')

    # log directory 생성
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    # Tensorboard 정의
    writer = SummaryWriter(log_dir=log_dir)

    # 모델 콜
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder()
    model = model.to(device)

    # 손실 함수와 옵티마이저
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    


    # 모델 학습
    global_step = 0
    num_fold = 5
    total_error = []
    val_error_ep = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(raw_stft)): # raw_stft의 shape: (1613, 128, 128, 3)
        if fold == 1: break

        print(f'Fold {fold + 1}/{num_fold}')

        train_dataset = STFTDataLoader(image=raw_stft[train_indices], mode='train', transform=train_transform)
        val_dataset = STFTDataLoader(image=raw_stft[val_indices], mode='val', transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        total_steps = len(train_loader)
        for epoch in range(1, num_epochs+1):
            # if epoch == 5: break
            training_loss = 0
            model.train()
            for step, sample in enumerate(train_loader):
                # if step == 5: break
                optimizer.zero_grad()

                sample = torch.tensor(sample, dtype=torch.float32, device=device)
                prediction = model(sample)
                train_loss = criterion(prediction, sample)
                
                for param_group in optimizer.param_groups:
                    current_lr = learning_rate * ((1 - epoch / num_epochs) ** 0.9)
                    param_group['lr'] = current_lr
                
                train_loss.backward()
                optimizer.step()

                training_loss += train_loss.item()
                
                # training_loss += train_loss.item()    # training_loss = training_loss + loss.item()
                # if (step % 10 == 0):
                #     avg_train_loss = training_loss/10
                # elif (step == len(train_loader)-1):
                #     avg_train_loss = training_loss / (step % 10)

                if step % 10 == 0:
                    train_image = torchvision.utils.make_grid(sample)
                    writer.add_scalar("Training/Loss", training_loss, epoch)
                    writer.add_scalar("Training/Learning Rate", learning_rate, epoch)
                    writer.add_image('Training/Input_Image/', train_image, global_step=global_step)


                    sys.stdout.write(f"\rEpoch: {epoch} \t | step: {step+1}/{total_steps} \t | Average Train Loss: {training_loss:.4f}")
                    sys.stdout.flush()
                    time.sleep(0)
            print()



            model.eval()
            with torch.no_grad():
                avg_val_loss = 0
                errors = []
                for num, val_sample in enumerate(val_loader):
                    val_sample = val_sample.to(device)

                    val_prediction = model(val_sample)

                    val_loss = criterion(val_prediction, val_sample)
                    avg_val_loss += val_loss.item()

                    pred = torchvision.utils.make_grid(val_prediction)
                    # writer.add_scalar("Validation/Loss", val_loss, epoch)
                    writer.add_image("Validation/Reconstructed_Image", pred, epoch)

                    # Mean Absolute Error (MAE)
                    error = torch.mean(torch.abs(val_prediction - val_sample), axis=0)
                    error = torch.mean(error)
                    errors.append(error.cpu().numpy())

                errors = np.mean(np.array(errors))
                total_error.append(errors)
                
                avg_val_loss = avg_val_loss / len(val_loader)
                writer.add_scalar("Validation/Loss", avg_val_loss, epoch)

            print()
            
            val_error_ep.append(error)
            
            global_step += 1
            if len(val_error_ep) > 1 and (val_error_ep[-1] < min(val_error_ep[:-1])):
                sys.stdout.write(f"\rValidation Result: Average Val Loss: {avg_val_loss:.4f}")
                sys.stdout.flush()
                
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            f'{log_dir}/CAE_checkpoint_{epoch}.pth')
        
    print('Finished Training')



def get_pred_label(model_pred, t):
    # (0:정상, 1:불량)로 Label 변환
    model_pred = np.where(model_pred <= t, 0, model_pred)
    model_pred = np.where(model_pred > t, 1, model_pred)
    return model_pred



def test():
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    pth = 'CAE_checkpoint_88.pth'
    test_folder = 'stft_try2'
    
    raw_stft = np.load('./DATASET/stft_image_dataset.npy')
    test_stft = np.load(osp.join(os.getcwd(), 'DATASET', 'stft_image_test_dataset.npy'))


    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
    ])
    test_dataset = STFTDataLoader(test_stft, mode='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(osp.join(os.getcwd(), 'logs', test_folder, pth), map_location=device)

    model = ConvAutoencoder()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()

    test_scores = []
    # latent_vectors = []
    for num, test_image in enumerate(test_dataloader):
        test_image = torch.tensor(test_image, dtype=torch.float32, device=device)
        
        test_pred = model(test_image) # model prediction
        # z = model.encoder(test_image)
        
        # latent_vectors.append(z.cpu().detach().numpy())
        
        # MAE
        # error = torch.abs(test_image - test_pred)
        # score = torch.mean(torch.mean(error.squeeze(), axis=0))
        # mean_scores.append(score.detach().cpu().item())
        error = torch.mean(torch.abs(test_pred - test_image), axis=(1, 2, 3))
        # error = torch.mean(error)
        # train_score.append(error.cpu().item())
        test_scores.append(error.detach().cpu().numpy())
    
    test_scores = np.concatenate(test_scores, axis=0)
    # latent_vectors = np.concatenate(latent_vectors, axis=0)
    # # 잠재 벡터를 2차원으로 축소하여 시각화
    # # tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    # tsne = TSNE(n_components=2, random_state=0)
    # latent_tsne = tsne.fit_transform(latent_vectors.reshape(1510, -1))

    # # 시각화
    # plt.figure(figsize=(10, 8))
    # plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], cmap='viridis')
    # plt.colorbar()
    # plt.xlabel('TSNE Dimension 1')
    # plt.ylabel('TSNE Dimension 2')
    # plt.title('Latent Space Visualization')
    # plt.savefig('./test_tsne.png')
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(128, 128))

    ])
    # kf = KFold(n_splits=5, shuffle=True)
    # for fold, (train_indices, val_indices) in enumerate(kf.split(raw_stft)): # raw_stft의 shape: (1613, 128, 128, 3)
    #     if fold == 1: break
        
    train_dataset = STFTDataLoader(image=raw_stft, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    
    train_score = []
    train_latent_vectors = []
    for step, sample in enumerate(train_loader):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
        prediction = model(sample)
        
        # z = model.encoder(sample)
        # train_latent_vectors.append(z.cpu().detach().numpy())
        
        # Mean Absolute Error (MAE)
        error = torch.mean(torch.abs(prediction - sample), axis=(1, 2, 3))
        # error = torch.mean(error)
        # train_score.append(error.cpu().item())
        # error = torch.mean(error)
        train_score.append(error.detach().cpu().numpy())
        # train_score.append(error.detach().cpu().item())
        
        
    # train_latent_vectors = np.concatenate(train_latent_vectors, axis=0)
    # # 잠재 벡터를 2차원으로 축소하여 시각화
    # # tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    # tsne = TSNE(n_components=2, random_state=0)
    # latent_tsne = tsne.fit_transform(train_latent_vectors.reshape(1277, -1))

    # # 시각화
    # plt.figure(figsize=(10, 8))
    # plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], cmap='viridis')
    # plt.colorbar()
    # plt.xlabel('TSNE Dimension 1')
    # plt.ylabel('TSNE Dimension 2')
    # plt.title('Latent Space Visualization')
    # # plt.show()
    # plt.savefig('./train_tsne.png')
    
    train_score = np.concatenate(train_score, axis=0)
    # train_score = np.array(train_score)
    t = max(train_score)
    result = get_pred_label(model_pred=test_scores, t=t)
    print(Counter(result))

    submit = pd.read_csv('./DATASET/sample_submission.csv')
    submit['LABEL'] = result
    print(submit.head())
    submit.to_csv('./DATASET/summit.csv', index=False)


if __name__ == "__main__":
    # train()
    test()