global_step = 0
num_fold = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = [] #JYH3-1 과제용 
for fold, (train_indices, val_indices) in enumerate(kf.split(raw_stft)):
    #if fold == 1: break
    print(f'Fold {fold + 1}/{num_fold}')

    train_dataset = STFTDataLoader(image=raw_stft[train_indices], mode='train', transform=train_transform)
    val_dataset = STFTDataLoader(image=raw_stft[val_indices], mode='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    total_steps = len(train_loader)

    for epoch in range(1, epochs+1):
        # if epoch == 5: break
        training_loss = 0
        model.train()
        for step, sample in enumerate(train_loader):

            #if step == 5: break
            optimizer.zero_grad()

            sample = torch.tensor(sample, dtype=torch.float32, device=device)
            prediction = model(sample)
            train_loss = criterion(prediction, sample)

            train_loss.backward()
            optimizer.step()

            training_loss += train_loss.item()    # training_loss = training_loss + loss.item()
            if (step % 10 == 0):
                avg_train_loss = training_loss/10
            elif (step == len(train_loader)-1):
                avg_train_loss = training_loss / (step % 10)


            train_image = torchvision.utils.make_grid(sample)
            #writer.add_scalar("Training/Loss", avg_train_loss, epoch)
            #writer.add_image('Training/Input_Image/', train_image, global_step=global_step)

            sys.stdout.write(f"\rEpoch: {epoch} \t | step: {step+1}/{total_steps} \t | Average Train Loss: {avg_train_loss:.4f}")
            sys.stdout.flush()
            time.sleep(0)
        print()

        model.eval()
        with torch.no_grad(): #autograd engine을 꺼버린다.
            avg_val_loss = 0
            score_arr = []

            for num, val_sample in enumerate(val_loader):
                val_sample = val_sample.to(device)
                # JYH3-1 시작
                encoded  = model.encoder(val_sample)
                b = np.array(encoded.view(encoded.size(0), -1).cpu().numpy())
                features.append(b)
                # JYH3-1
                val_prediction = model(val_sample)
                val_loss = criterion(val_prediction, val_sample)
                avg_val_loss += val_loss.item()

                pred = torchvision.utils.make_grid(val_prediction)
                #jyh writer.add_scalar("Validation/Loss", val_loss, epoch)
                #jyh writer.add_image("Validation/Reconstructed_Image", pred, epoch)

                # Mean Absolute Error (MAE)
                score = torch.mean(torch.abs(val_prediction - val_sample), axis=1)
                score_arr.append(score.cpu().numpy())

                # # SSIM 계산
                # val_prediction_np = val_prediction.permute(0, 2, 3, 1).cpu().numpy()
                # val_sample_np = val_sample.permute(0, 2, 3, 1).cpu().numpy()
                # ssim_score = 0
                # for orscore_arrigin, pred in zip(val_sample_np, val_prediction_np):
                #     origin = np.dot(origin[..., :3], [0.299, 0.587, 0.114])
                #     pred = np.dot(pred[..., :3], [0.299, 0.587, 0.114])
                #     score = ssim(pred, origin, win_size=7)
                #     ssim_score += score

                # batch_ssim_score = ssim_score / batch_size
                # sample_ssim_score += batch_ssim_score


            #sample_ssim_score = sample_ssim_score / len(val_loader)
            score_arr = np.array(score_arr)
            avg_val_loss = avg_val_loss / len(val_loader)
            #sys.stdout.write(f"\rValidation Result: Average Val Loss: {avg_val_loss:.4f} \t | Average ssim score: {sample_ssim_score:.4f}")
            sys.stdout.write(f"\rValidation Result: Average Val Loss: {avg_val_loss:.4f}")
            sys.stdout.flush()

            #writer.add_scalar("Validation/Loss", avg_val_loss, epoch)
            # writer.add_scalar("Validation/ssim score", sample_ssim_score, epoch)

        print()

        global_step += 1
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f'{log_dir}/CAE_checkpoint_{epoch}.pth')

features = np.concatenate(features, axis=0) 
print('======= Finished Training =======')

# JYH3-1 과제용 Apply TSNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca')
tsne_features = tsne.fit_transform(features)
plt.figure(figsize=(10, 6))
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], cmap='viridis') 
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='STFT')
plt.show() #JYH3-1 과제용 끝
