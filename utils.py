import os
from nonconf_scores import *
np.random.seed(55)


def validate_config(cal_output, cal_target, config):
    errors = []
    # Check DATASET_NAME
    if not isinstance(config.get("DATASET_NAME"), str):
        errors.append("DATASET_NAME should be a string.")
    for part in ['cal_outputs','test_outputs','cal_target','test_target']:
        p=os.path.join(os.getcwd(), 'outputs', f'{part}_{config["DATASET_NAME"]}.npy')
        if not os.path.isfile(p):
            errors.append(f"Can't find file {p}")
    # Check ALPHA
    data_accuracy = min([(head.argmax(axis=1) == cal_target.astype(int)).mean() for head in  cal_output])
    max_alpha = 1-max(0, data_accuracy)
    if not isinstance(config.get("ALPHA"), float) or not (0 <= config["ALPHA"] <= max_alpha):
        errors.append(f"ALPHA should be a float between 0 and {max_alpha:.4f}.")
    # Check b
    if not isinstance(config.get("b"), int) or config["b"] < 1:
        errors.append("b should be an integer >= 1.")
    # Check N_HEADS
    if not isinstance(config.get("N_HEADS"), int) or config["N_HEADS"] < 1:
        errors.append("N_HEADS should be an integer >= 1.")
    elif config["N_HEADS"] > cal_output.shape[0]:
        errors.append(f"N_HEADS should be at most {cal_output.shape[0]}.")
    # Check SCORING_METHOD
    if not isinstance(config.get("SCORING_METHOD"), str):
        errors.append("SCORING_METHOD should be a string.")
    if config.get("SCORING_METHOD") not in ['RAPS','APS','SAPS','NAIVE']:
        errors.append("SCORING_METHOD should be one of ['RAPS','APS','SAPS','NAIVE'].")
    # Raise all errors at the end
    if errors:
        raise ValueError("\n".join(errors))


def create_true_rest_sets(scores: np.ndarray,target: np.ndarray,stage: str)-> (tuple,tuple):
    """
    inputs:
    scores: shape ( num heads, num_samples,num_class)
    targets (np.ndarray of floats): shape( num_samples)
    outputs:
    true_preds : tuple of size  1 or num_class for ( CAL, Val) equivalently, with np.ndarray shape(num_heads, num_samples) as elemnts of the tuple
    rest_preds_concat : tuple of size  1 or num_class for ( CAL, Val) equivalently, with np.ndarray shape(num_heads, num_samples*(num_class-1)) as elemnts of the tuple
    """
    rest_preds=np.zeros([scores.shape[0],scores.shape[1],scores.shape[2]-1])
    true_preds=np.zeros([scores.shape[0],scores.shape[1]])
    if stage=='CAL':
        for head in range(scores.shape[0]):
            for row in range(scores.shape[1]):
                selected_elements=np.delete(scores[head,row,:], int(target[row]), axis=0)
                rest_preds[head,row,:]=selected_elements
            true_preds[head]=scores[head,:,:][np.arange(len(scores[head,:,:])), target.astype(int)]
        rest_preds_concat =rest_preds
    elif stage=='VAL':
        results = [(scores[:, :, cl], np.delete(scores, cl, axis=2)) for cl in range(scores.shape[2])]
        true_preds, rest_preds_concat = zip(*results)

    return true_preds, rest_preds_concat
def create_random_split(cal_output, cal_target, test_output, test_target):
    big_output = np.concatenate((cal_output, test_output), axis=1)
    big_target = np.concatenate((cal_target, test_target), axis=0)
    num_cal_samples = len(cal_target)
    num_test_samples = len(test_target)
    indices = np.arange(num_cal_samples + num_test_samples)
    np.random.shuffle(indices)
    cal_indices = indices[:num_cal_samples]
    test_indices = indices[num_cal_samples:]
    cal_output_new = big_output[:, cal_indices, :]
    cal_target_new = big_target[cal_indices]
    test_output_new = big_output[:, test_indices, :]
    test_target_new = big_target[test_indices]
    return cal_output_new, cal_target_new, test_output_new, test_target_new
def load_data(config):
    cal_output = np.load(os.path.join(os.getcwd(), 'outputs', f'cal_outputs_{config["DATASET_NAME"]}.npy'))
    cal_target = np.load(os.path.join(os.getcwd(), 'outputs', f'cal_target_{config["DATASET_NAME"]}.npy'))
    test_output = np.load(os.path.join(os.getcwd(), 'outputs', f'test_outputs_{config["DATASET_NAME"]}.npy'))
    test_target = np.load(os.path.join(os.getcwd(), 'outputs', f'test_target_{config["DATASET_NAME"]}.npy'))
    return cal_output,cal_target,test_output,test_target
def create_scores(data,config):
    return  np.squeeze([get_scoring_method(config['SCORING_METHOD'])(data[idxhead], 'SCORES')
                             for idxhead in range(len(data))], axis=1)
def generate_cal_opt_sets(cal_scores,cal_target_new):
    precal_scores, opt_scores = cal_scores[:, :cal_scores.shape[1] // 2, :], cal_scores[:,
                                                                             cal_scores.shape[1] // 2:, :]
    precal_target, opt_target = cal_target_new[:cal_target_new.shape[0] // 2], cal_target_new[
                                                                               cal_target_new.shape[0] // 2:]
    return precal_scores,precal_target,opt_scores,opt_target

def get_scoring_method(method_name):
    if method_name=='RAPS':
        return conf_score_RAPS
    elif method_name=='SAPS':
        return conf_score_SAPS
    elif method_name=='NAIVE':
        return conf_score_NAIVE
    elif method_name=='APS':
        return conf_score_APS
    raise ValueError("Scoring method is not recognized, choose one of RAPS, SAPS , NAIVE , APS")