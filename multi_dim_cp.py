from load_config import load_config
from scipy.spatial import KDTree
from utils import load_data, create_scores, generate_cal_opt_sets, create_true_rest_sets, create_random_split, \
    validate_config
from nonconf_scores import *








# def find_closest_columns(array1, array2):
#     """
#     Find the closest column in array1 for each column in array2 using KDTree for efficiency.
#
#     Parameters:
#     array1 (numpy.ndarray): Array of shape (n_dim, n_samples)
#     array2 (numpy.ndarray): Array of shape (n_dim, n_samples2)
#
#     Returns:
#     list: List of indices of the closest columns in array1 for each column in array2
#     """
#     tree = KDTree(array1.T)
#     closest_indices = tree.query(array2.T)[1]
#     return np.array(closest_indices)
#
# def create_wanted_bins_centers_part_1(cal_true_preds, cal_rest_preds, alpha):
#     n_dims = cal_true_preds.shape[0]
#     cal_rest_flat = cal_rest_preds.reshape(n_dims, -1)
#     centers_unique, count_cal_points_in_center = np.unique(cal_true_preds, axis=1, return_counts=True)
#     closest_center_idx_to_rest_points = find_closest_columns(centers_unique,cal_rest_flat)
#     closest_center_idx_to_rest_points_unique,count_of_rest_points_belongs_to_center_idx_unique=np.unique(closest_center_idx_to_rest_points, return_counts=True)
#     total_score = np.ones(count_cal_points_in_center.shape)
#     total_score[closest_center_idx_to_rest_points_unique] = +count_of_rest_points_belongs_to_center_idx_unique
#     total_score = total_score / count_cal_points_in_center
#     wanted_centers,alpha_hat=create_wanted_bins_centers_part_2(total_score, count_cal_points_in_center, centers_unique,alpha)
#     return wanted_centers,centers_unique,alpha_hat,count_cal_points_in_center,total_score
#
# def  create_wanted_bins_centers_part_2(total_score, count_cal_points_in_center, centers_unique,alpha):
#     n_samples=total_score.shape[0]
#     alpha_hat = (n_samples + 1) * (1 - alpha)/ n_samples
#     ordered_indexes = np.argsort(total_score)
#     wanted_centers = np.arange(centers_unique.shape[1])[ordered_indexes][
#         count_cal_points_in_center[ordered_indexes].cumsum() <= np.ceil(alpha_hat * n_samples)]
#     return wanted_centers,alpha_hat
#
#
#
# def allocate_points(precal_scores, precal_target,opt_scores,opt_target,test_scores,test_target,alpha):
#     cal_true_preds, cal_rest_preds = create_true_rest_sets(precal_scores, precal_target, 'CAL')
#     wanted_coverage_achived=False
#     alpha_init=alpha
#     wanted_centers, cal_true_preds_with_zero, alpha_hat, count_cal_points_in_center, total_score = create_wanted_bins_centers_part_1(cal_true_preds, cal_rest_preds, alpha_init)
#     closest_opt_points_idxs = find_closest_columns(cal_true_preds_with_zero, opt_scores.reshape(opt_scores.shape[0], -1))
#     closest_test_points_idxs = find_closest_columns(cal_true_preds_with_zero, test_scores.reshape(test_scores.shape[0], -1))
#     coverage, mean_set, _ = test_bins(opt_scores, opt_target, wanted_centers, cal_true_preds_with_zero,closest_opt_points_idxs)
#     ordered_indexes = np.argsort(total_score)
#     if coverage >= 1 - alpha:
#         wanted_coverage_achived = True
#     while not wanted_coverage_achived:
#         wanted_centers = np.arange(count_cal_points_in_center.shape[0])[ordered_indexes][:min(count_cal_points_in_center.shape[0],wanted_centers.shape[0]+1)]
#         coverage,mean_set,_=test_bins(opt_scores, opt_target, wanted_centers, cal_true_preds_with_zero,closest_opt_points_idxs)
#         if coverage>=1-alpha:
#             wanted_coverage_achived=True
#     coverage, mean_set ,result_mat= test_bins(test_scores, test_target, wanted_centers, cal_true_preds_with_zero,closest_test_points_idxs)
#
#     return coverage,mean_set,result_mat
#
# def test_bins(test_score, test_target, wanted_centers, cal_true_preds_with_zero,closest_test_points_idxs='None'):
#     n_dims = test_score.shape[0]
#     n_samples = test_score.shape[1]
#     n_labels = test_score.shape[2]
#     test_flat = test_score.reshape(n_dims, -1)
#     limits = np.max(cal_true_preds_with_zero[:, wanted_centers], axis=1)
#     ommit_outliers = ~np.any(test_score > limits[:, np.newaxis, np.newaxis], axis=0)
#     if type(closest_test_points_idxs)==str:
#         closest_test_points_idxs = find_closest_columns(cal_true_preds_with_zero, test_flat)
#     result_mat = np.isin(closest_test_points_idxs, wanted_centers).reshape(n_samples, n_labels)
#     result_mat = result_mat * ommit_outliers
#     zero_rows=result_mat.sum(axis=1) == 0
#     if zero_rows.any():
#         flaten_zero_columns=test_score[:,zero_rows, :].reshape(n_dims,-1)
#         tree_zeros = KDTree(cal_true_preds_with_zero[:,wanted_centers].T)
#         distances,closest_zero_indices = tree_zeros.query(flaten_zero_columns.T)
#         force_one_idx=np.argmin(distances.reshape(-1, n_labels), axis=1)
#         result_mat[zero_rows, force_one_idx]=True
#     return result_mat[range(len(result_mat)), test_target.astype(int)].sum() / len(result_mat), result_mat.sum(
#         axis=1).mean(),result_mat


def find_closest_columns(array1, array2,phase,config):
    """
    Find the closest column in array1 for each column in array2 using KDTree for efficiency.

    Parameters:
    array1 (numpy.ndarray): Array of shape (n_dim, n_samples)
    array2 (numpy.ndarray): Array of shape (n_dim, n_samples2)

    Returns:
    list: List of indices of the closest columns in array1 for each column in array2
    """
    tree = KDTree(array1.T)
    if phase=='cal':
        closest_indices = tree.query(array2.T)[1]
    elif phase == 'test':
     closest_indices = tree.query(array2.T,k=config['b'])[1]
    return np.array(closest_indices)

def create_wanted_bins_centers_part_1(cal_true_preds, cal_rest_preds, alpha,config):
    n_dims = cal_true_preds.shape[0]
    cal_rest_flat = cal_rest_preds.reshape(n_dims, -1)
    centers_unique, count_cal_points_in_center = np.unique(cal_true_preds, axis=1, return_counts=True)
    closest_center_idx_to_rest_points = find_closest_columns(centers_unique,cal_rest_flat,'cal',config)
    closest_center_idx_to_rest_points_unique,count_of_rest_points_belongs_to_center_idx_unique=np.unique(closest_center_idx_to_rest_points, return_counts=True)
    total_score = np.ones(count_cal_points_in_center.shape)
    total_score[closest_center_idx_to_rest_points_unique] = +count_of_rest_points_belongs_to_center_idx_unique
    total_score = total_score / count_cal_points_in_center
    wanted_centers,alpha_hat=create_wanted_bins_centers_part_2(total_score, count_cal_points_in_center, centers_unique,alpha)
    return wanted_centers,centers_unique,alpha_hat,count_cal_points_in_center,total_score

def  create_wanted_bins_centers_part_2(total_score, count_cal_points_in_center, centers_unique,alpha):
    n_samples=total_score.shape[0]
    alpha_hat = (n_samples + 1) * (1 - alpha)/ n_samples
    ordered_indexes = np.argsort(total_score)
    wanted_centers = np.arange(centers_unique.shape[1])[ordered_indexes][
        count_cal_points_in_center[ordered_indexes].cumsum() <= np.ceil(alpha_hat * n_samples)]
    return wanted_centers,alpha_hat



def allocate_points(precal_scores, precal_target,opt_scores,opt_target,test_scores,test_target,alpha,config):
    cal_true_preds, cal_rest_preds = create_true_rest_sets(precal_scores, precal_target, 'CAL')
    wanted_coverage_achived=False
    alpha_init=alpha
    wanted_centers, cal_true_preds_with_zero, alpha_hat, count_cal_points_in_center, total_score = create_wanted_bins_centers_part_1(cal_true_preds, cal_rest_preds, alpha_init,config)
    closest_opt_points_idxs = find_closest_columns(cal_true_preds_with_zero, opt_scores.reshape(opt_scores.shape[0], -1),'test',config)
    closest_test_points_idxs = find_closest_columns(cal_true_preds_with_zero, test_scores.reshape(test_scores.shape[0], -1),'test',config)
    coverage, mean_set, _ = test_bins(opt_scores, opt_target, wanted_centers, cal_true_preds_with_zero,closest_opt_points_idxs,config)
    ordered_indexes = np.argsort(total_score)
    if coverage >= 1 - alpha:
        wanted_coverage_achived = True
    while not wanted_coverage_achived:
        wanted_centers = np.arange(count_cal_points_in_center.shape[0])[ordered_indexes][:min(count_cal_points_in_center.shape[0],wanted_centers.shape[0]+1)]
        coverage,mean_set,_=test_bins(opt_scores, opt_target, wanted_centers, cal_true_preds_with_zero,closest_opt_points_idxs,config)
        if coverage>=1-alpha:
            wanted_coverage_achived=True
    coverage, mean_set ,result_mat= test_bins(test_scores, test_target, wanted_centers, cal_true_preds_with_zero,closest_test_points_idxs,config)
    return coverage,mean_set,result_mat,(wanted_centers,cal_true_preds_with_zero)

def test_bins(test_score, test_target, wanted_centers, cal_true_preds_with_zero,closest_test_points_idxs,config):
    n_dims = test_score.shape[0]
    n_samples = test_score.shape[1]
    n_labels = test_score.shape[2]
    limits = np.max(cal_true_preds_with_zero[:, wanted_centers], axis=1)
    ommit_outliers = ~np.any(test_score > limits[:, np.newaxis, np.newaxis], axis=0)
    scores_in = np.isin(closest_test_points_idxs, wanted_centers)
    scores_in = np.expand_dims(scores_in, axis=1) if scores_in.ndim == 1 else scores_in
    result_mat = (scores_in.sum(axis=1) >= max(config['b'] * 0.5, 1)).reshape(n_samples,n_labels)
    result_mat = result_mat * ommit_outliers
    zero_rows=result_mat.sum(axis=1) == 0
    if zero_rows.any():
        flaten_zero_columns=test_score[:,zero_rows, :].reshape(n_dims,-1)
        tree_zeros = KDTree(cal_true_preds_with_zero[:,wanted_centers].T)
        distances,closest_zero_indices = tree_zeros.query(flaten_zero_columns.T)
        force_one_idx=np.argmin(distances.reshape(-1, n_labels), axis=1)
        result_mat[zero_rows, force_one_idx]=True

    return result_mat[range(len(result_mat)), test_target.astype(int)].sum() / len(result_mat), result_mat.sum(
        axis=1).mean(),result_mat


def run(config):
    cal_output,cal_target,test_output,test_target=load_data(config)
    validate_config(cal_output,cal_target,config)

    cal_output_new, cal_target_new, test_output_new, test_target_new = create_random_split(cal_output[:config['N_HEADS'], :, :], cal_target, test_output[:config['N_HEADS'], :, :], test_target)

    cal_scores =create_scores(cal_output_new,config)
    test_scores =create_scores(test_output_new,config)

    precal_scores,precal_target,opt_scores,opt_target=generate_cal_opt_sets(cal_scores, cal_target_new)

    coverage, mean_set,prediction_sets_new_meth,_ = allocate_points(precal_scores, precal_target,opt_scores, opt_target, test_scores,test_target_new, config['ALPHA'],config)
    print(f"\n################################## Results ##################################\n")
    print(f"Coverage archived: {coverage}\n")
    print(f"Coverage wanted: {1-config['ALPHA']}\n")
    print(f"Mean-Predicted-Set-Size: {mean_set}\n")





if __name__=="__main__":
    config_name='multi_dim_cp_config'
    config=load_config(config_name)
    run(config)




