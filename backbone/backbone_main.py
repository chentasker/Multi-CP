from backbone_model import *
from criterions_and_optimizers import *
from training_page import *
from scipy.special import softmax
from load_config import *


def main(current_config):
    best_val_loss = np.ones(config['general']['heads_num'])*np.inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, cal_loader=load_data(current_config)
    model = ModifiedResNet50(current_config).to(device)

    optimizer = set_optimizer(model,current_config)
    criterion = set_criterion(current_config)

    if current_config['current_step']['training_phase'][0]=="retrain" or current_config['current_step']['training_phase'][0]=="eval":
        model,_=load_best_weights(model, optimizer, config['general']['save_path'])
    if not current_config['current_step']['training_phase'][0]=="eval":
        model,best_val_loss=training_loop(model,train_loader,val_loader,optimizer,criterion,device,best_val_loss,current_config)
    test_loss, test_accuracy,test_output,test_targets = evaluate_individual_heads(model, test_loader, criterion, device)
    cal_loss, cal_accuracy,cal_output,cal_targets = evaluate_individual_heads(model, cal_loader, criterion, device)
    print(f"Cal average accuracy:{np.array(cal_accuracy).mean()}")
    print(f"Test average accuracy:{np.array(test_accuracy).mean()}")
    print(f"Train sampels:{len(train_loader.dataset)}\nVal sampels:{len(val_loader.dataset)}\nTest sampels:{test_targets.shape[0]}\nCAL sampels:{cal_targets.shape[0]}")

    if config['general']['save_output'][0] and current_config['current_step']['training_phase'][0]=="eval":
        save_dir = config['general']['save_output'][1]
        save_name=config['general']['save_output'][2]
        np.save(f"{save_dir}/test_outputs_{config['general']['dataset_name']}_{save_name}.npy", softmax(test_output,axis=2))
        np.save(f"{save_dir}/cal_outputs_{config['general']['dataset_name']}_{save_name}.npy", softmax(cal_output,axis=2))
        np.save(f"{save_dir}/cal_target_{config['general']['dataset_name']}_{save_name}.npy", cal_targets)
        np.save(f"{save_dir}/test_target_{config['general']['dataset_name']}_{save_name}v.npy", test_targets)



    print(f"Test Loss: {np.array(test_loss).mean()}, Test Accuracy: {np.array(test_accuracy).mean()}")


if __name__ == "__main__":
    config=load_config()
    current_config={}

    current_config['current_step']={}
    current_config['current_step']['batch_size']= config['general']['batch_size']
    current_config['current_step']['dropout_prob']= config['general']['dropout_prob']
    current_config['current_step']['num_layers']= config['general']['num_layers']
    current_config['current_step']['optimizer_name']=config['general']['optimizer_name']
    current_config['current_step']['weight_decay'] = config['general']['weight_decay']

    for step_index, s in enumerate(config['training_plan'].values()):
        current_config['current_step']['trained_heads'] = s['trained_heads']
        current_config['current_step']['training_phase'] = s['training_phase']
        current_config['current_step']['num_epochs'] = s['num_epochs']
        current_config['current_step']['criterion_name'] = s['criterion_name']
        current_config['current_step']['lr_features']= s['lr_features']
        current_config['current_step']['lr_heads'] = s['lr_heads']
        print(current_config['current_step'])
        main(current_config)







