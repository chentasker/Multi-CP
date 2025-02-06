from tqdm import tqdm
from data_and_weights import *


config = load_config()

def weights_switcher(model, current_config, num_header, state):
    if current_config['current_step']['training_phase'][1] == "whole":
        for param in model.features.parameters():
            param.requires_grad = state[0]  # Freeze/unfreeze feature extractor
        model.features.train() if state[1] else model.features.eval()

    for h_idx in num_header:
        model.classification_heads[h_idx].train() if state[1] else model.classification_heads[h_idx].eval()
        for param in model.classification_heads[h_idx].parameters():
            param.requires_grad = state[0]  # Freeze/unfreeze classification heads

    return model



def training_loop(model,train_loader,test_loader,optimizer,criterion,device,best_val_loss,current_config):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_config['current_step']['num_epochs'], eta_min=1e-6)
    val_loss_l ,train_loss_l=[],[]
    val_accu_l =[]
    model.eval()
    if current_config['current_step']['trained_heads'][0]=='all':
        num_header = range(config['general']['heads_num'])
    else:
        num_header=current_config['current_step']['trained_heads']
    model=weights_switcher(model, current_config, num_header, [True, True])
    for epoch in range(current_config['current_step']['num_epochs']):
        print(f"\nEpoch {epoch+1} out of {current_config['current_step']['num_epochs']}")

        train_loss,model= train(model, train_loader, optimizer, criterion, device,num_header)
        val_loss, val_accuracy,_,_  = evaluate_individual_heads(model, test_loader, criterion, device)

        if val_loss[num_header].mean() < best_val_loss[num_header].mean():
            best_val_loss[num_header] = val_loss[num_header]
            if config['general']['save_weights']:
                save_best_weights(model, optimizer, epoch, best_val_loss, config['general']['save_path'])

        # if np.mean(val_loss)< best_val_loss:
        #     best_val_loss = np.mean(val_loss)
        #     if  config['general']['save_weights']:
        #         save_best_weights(model, optimizer, epoch, best_val_loss, config['general']['save_path'])

        print(f"\nEpoch {epoch + 1}/{current_config['current_step']['num_epochs']}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss} \n"
              f"Val Accuracy: {val_accuracy}")
        scheduler.step(torch.tensor(val_loss).mean())
        val_loss_l.append(val_loss),train_loss_l.append(train_loss),val_accu_l.append(val_accuracy)
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")
    model=weights_switcher(model, current_config, num_header, [False, False])
    return model,best_val_loss


def train(model, train_loader, optimizer, criterion, device,num_header):
    running_loss = 0.0
    for data, target in tqdm(train_loader,desc="Batch Progress",ascii=True):

        data, target = data.to(device), target.to(device)
        total_loss=torch.zeros(1).to(device)
        optimizer.zero_grad()

        # Get the outputs from all heads
        all_outputs = model(data)
        for outputs_index in num_header:
            loss = criterion(all_outputs, target.long(),outputs_index)
            total_loss += loss
        total_loss.backward()

        optimizer.step()
        running_loss += total_loss.item()

    return running_loss / len(train_loader),model

def evaluate_individual_heads(model, data_loader, criterion, device):
    model.eval()
    running_losses = np.zeros(config['general']['heads_num'])

    corrects = [0] *config['general']['heads_num']
    results=[]
    count=-1
    targets=[]
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            all_outputs = model(data)
            for i,outputs in enumerate(all_outputs):
                loss = criterion(all_outputs, target.long(),i)
                running_losses[i] += loss.item()
                _, predicted = all_outputs[i].max(1)
                corrects[i] += predicted.eq(target).sum().item()
                if count == -1:
                    results.append(outputs.cpu().numpy())
                else:
                    results[i] = np.concatenate([results[i], outputs.cpu().numpy()], axis=0)
            count = 0
            targets = np.concatenate([targets, target.cpu().numpy()], axis=0)

    losses =  np.array(running_losses) / len(data_loader)
    accuracies = (np.array(corrects)/results[0].shape[0]).tolist()
    return losses, accuracies,np.array(results),targets




