from load_config import load_config
from scipy.spatial import KDTree
from utils import load_data, compute_scores, generate_Dcal_Dcells_sets, create_true_rest_sets, create_random_split, \
    validate_config
from nonconf_scores import *


def find_closest_columns(array1, array2, phase,config):
    """
        Find the b nearest cell centers (Dcells_true) for each point in array2.
        creates points cloud around each center that defines the cell.

        Parameters:
            array1 (numpy.ndarray): Array of shape (n_dim, n_samples) representing the reference cell centers.
            array2 (numpy.ndarray): Array of shape (n_dim, n_samples2) representing the query points.

        Returns:
            np.ndarray: Indices of the b closest columns in array1 for each column in array2.
    """
    tree = KDTree(array1.T)
    if phase == 'cal':
        closest_indices = tree.query(array2.T)[1]
    elif phase == 'test':
     closest_indices = tree.query(array2.T,k=config['b'])[1]
    return np.array(closest_indices)

def segment_S_and_rank_cells(Dcells_true : np.ndarray, Dcells_rest : np.ndarray, alpha : float, config : dict):
    """
        Segmenting S into cells according to Dcells true predictions and ranking the cells.
        Lines 4-7 of the algorithm are executed from here.

        Inputs:
            Dcells_true: The center of the cells Ci, shape (n_heads, n_Dcells_samples).
            Dcells_rest: The rest of the scores in the Dcells set, shape (n_heads, n_Dcells_samples, n_labels-1),
                         used in Eq. (9).
            alpha: 1 - desired coverage.
            config: The configuration of the algorithm's hyperparameters.

        Outputs:
            wanted_centers (np.ndarray): Selected cell indices (Ci).
            Dcells_true_unique (np.ndarray): All unique scores.
            alpha_hat: 1 - adjusted alpha value.
            count_Dcells_points_in_center (np.ndarray): Number of Dcells_true points in each center.
            total_score (np.ndarray): The score of each Di (calculated using Eq. (9)),
                                      which is used for ranking.
    """

    n_dims = Dcells_true.shape[0]
    cal_rest_flat = Dcells_rest.reshape(n_dims, -1)
    Dcells_true_unique, count_Dcells_points_in_center = np.unique(Dcells_true, axis=1, return_counts=True) # Remove duplicate centers (Line 6)
    # generate cells
    closest_center_idx_to_rest_points = find_closest_columns(Dcells_true_unique,cal_rest_flat,'cal',config)
    closest_center_idx_to_rest_points_unique,count_of_rest_points_belongs_to_center_idx_unique=np.unique(closest_center_idx_to_rest_points, return_counts=True)
    # calc score according to eq. (9)
    total_score = np.ones(count_Dcells_points_in_center.shape)
    total_score[closest_center_idx_to_rest_points_unique] += count_of_rest_points_belongs_to_center_idx_unique
    total_score = total_score / count_Dcells_points_in_center
    #Rank the cells and select initial selected cells (Line 7)
    wanted_centers,alpha_hat = calculate_psudo_selected_cells_idx(total_score, count_Dcells_points_in_center, Dcells_true_unique,alpha)
    return wanted_centers,Dcells_true_unique,alpha_hat,count_Dcells_points_in_center,total_score

def  calculate_psudo_selected_cells_idx(total_score : np.ndarray, count_Dcells_points_in_center : np.ndarray, Dcells_true_unique : np.ndarray,alpha : float):
    """
        Initialize an initial guess for η* (Line 9).

        Inputs:
            total_score (np.ndarray): The score of each Di (calculated using Eq. (9)), which is used for ranking.
            count_Dcells_points_in_center (np.ndarray): Number of Dcells_true points in each center.
            Dcells_true_unique (np.ndarray): All unique scores.
            alpha: 1 - desired coverage.

        Outputs:
            wanted_centers (np.ndarray): Selected cell indices (Ci).
            alpha_hat: Adjusted alpha value.
    """

    n_samples = total_score.shape[0]
    alpha_hat = (n_samples + 1) * (1 - alpha)/ n_samples  #fixing alpha value
    ordered_indexes = np.argsort(total_score)
    # Select centers according to rank until α̂ percent of Dcells_points_in_center is part of the chosen cells.
    wanted_centers = np.arange(Dcells_true_unique.shape[1])[ordered_indexes][
        count_Dcells_points_in_center[ordered_indexes].cumsum() <= np.ceil(alpha_hat * n_samples)]
    return wanted_centers,alpha_hat



def main_algo(Dcells_scores: np.ndarray, Dcells_target: np.ndarray, Dre_cal_scores: np.ndarray, Dre_cal_target: np.ndarray, test_scores: np.ndarray, test_target: np.ndarray, alpha: float, config :dict ):
    """
        This is the main part of the Soft Multi-Score Conformal Prediction Algorithm as presented in the article (B.2).
        Lines 4-16 of the algorithm are executed from here.

        Inputs:
            Dcells: The data on which the ranking will be performed.
                Dcells_score: Array of shape (n_dim, n_samples_Dcells, n_labels)
                Dcells_target: Array of shape (n_samples_Dcells,)

            Dre_cal: The data on which the cell selection will be performed.
                Dre_cal_scores: Array of shape (n_dim, n_samples_Dre_cal, n_labels)
                Dre_cal_target: Array of shape (n_samples_Dre_cal,)

            test: The unseen data during calibration on which the algorithm will be evaluated.
                test_scores: Array of shape (n_dim, n_samples_test, n_labels)
                test_target: Array of shape (n_samples_test,)

            alpha: 1 - desired coverage.

            config: The configuration of the algorithm's hyperparameters.

        Outputs:
            coverage: Achieved coverage on the test set.
            mean_set: Mean predicted set size on the test set.
            result_mat: A boolean array of shape (n_test_samples, n_labels), where each row represents
                        the predicted set per sample (True means the corresponding label is in the predicted set).
            (wanted_centers, Dcells_true_unique): Selected cells and all cells, used for illustrations.
    """

    Dcells_true, Dcells_rest = create_true_rest_sets(Dcells_scores, Dcells_target, 'CAL')
    wanted_coverage_achived=False
    alpha_init = alpha
    wanted_centers, Dcells_true_unique, alpha_hat, count_Dcells_points_in_center, total_score = segment_S_and_rank_cells(Dcells_true, Dcells_rest, alpha_init,config)
    closest_opt_points_idxs = find_closest_columns(Dcells_true_unique, Dre_cal_scores.reshape(Dre_cal_scores.shape[0], -1),'test',config)
    closest_test_points_idxs = find_closest_columns(Dcells_true_unique, test_scores.reshape(test_scores.shape[0], -1),'test',config)
    coverage, mean_set, _ = test_bins(Dre_cal_scores, Dre_cal_target, wanted_centers, Dcells_true_unique,closest_opt_points_idxs,config)
    ordered_indexes = np.argsort(total_score)
    if coverage >= 1 - alpha:
        wanted_coverage_achived = True
    while not wanted_coverage_achived:
        wanted_centers = np.arange(count_Dcells_points_in_center.shape[0])[ordered_indexes][:min(count_Dcells_points_in_center.shape[0],wanted_centers.shape[0]+1)]
        coverage,mean_set,_=test_bins(Dre_cal_scores, Dre_cal_target, wanted_centers, Dcells_true_unique,closest_opt_points_idxs,config)
        if coverage >= 1-alpha:
            wanted_coverage_achived = True
    coverage, mean_set ,result_mat = test_bins(test_scores, test_target, wanted_centers, Dcells_true_unique,closest_test_points_idxs,config)
    return coverage,mean_set,result_mat,(wanted_centers,Dcells_true_unique)

def test_bins(test_score : np.ndarray, test_target : np.ndarray, wanted_centers : np.ndarray, Dcells_true_unique : np.ndarray, closest_test_points_idxs : np.ndarray, config : dict):
    """
        Check whether the majority of the b closest neighbors of a label's score are in the selected cells.
        This function is used both in Lines 9-10 of the algorithm and during the evaluation phase.

        Inputs:
            test_scores (np.ndarray): Array of shape (n_dim, n_samples_test, n_labels),
                                      representing the test scores.
            test_target (np.ndarray): Array of shape (n_samples_test,), representing the true test labels.
            wanted_centers (np.ndarray): Selected cell indices (Ci).
            Dcells_true_unique (np.ndarray): All unique scores.
            closest_test_points_idxs (np.ndarray): Indices of the b closest cell centers.
            config (dict): Algorithm hyperparameters configuration.

        Outputs:
            coverage (float): Achieved coverage on the test set.
            mean_set (float): Mean predicted set size on the test set.
            result_mat (np.ndarray): A boolean array of shape (n_test_samples, n_labels), where each row
                                     represents the predicted set per sample (True means the corresponding
                                     label is in the predicted set).
    """

    n_dims = test_score.shape[0]
    n_samples = test_score.shape[1]
    n_labels = test_score.shape[2]
    limits = np.max(Dcells_true_unique[:, wanted_centers], axis=1)
    ommit_outliers = ~np.any(test_score > limits[:, np.newaxis, np.newaxis], axis=0)
    scores_in = np.isin(closest_test_points_idxs, wanted_centers)
    scores_in = np.expand_dims(scores_in, axis=1) if scores_in.ndim == 1 else scores_in
    result_mat = (scores_in.sum(axis=1) >= max(config['b'] * 0.5, 1)).reshape(n_samples,n_labels) #  Check majority voting (Line 9 and 15). If b=1 this is equvalent to the multi-score method
    result_mat = result_mat * ommit_outliers
    zero_rows=result_mat.sum(axis=1) == 0
    # deal with empty prediction sets
    if zero_rows.any():
        flaten_zero_columns=test_score[:,zero_rows, :].reshape(n_dims,-1)
        tree_zeros = KDTree(Dcells_true_unique[:,wanted_centers].T)
        distances,closest_zero_indices = tree_zeros.query(flaten_zero_columns.T)
        force_one_idx = np.argmin(distances.reshape(-1, n_labels), axis=1)
        result_mat[zero_rows, force_one_idx]=True

    return result_mat[range(len(result_mat)), test_target.astype(int)].sum() / len(result_mat), result_mat.sum(
        axis=1).mean(),result_mat


def run(config : dict):

    cal_output,cal_target,test_output,test_target = load_data(config) # Load the backbone's outputs (cal&test sets)
    validate_config(cal_output,cal_target,config)       # Validate the config file
    cal_output_new, cal_target_new, test_output_new, test_target_new = create_random_split(cal_output[:config['N_HEADS'], :, :], cal_target, test_output[:config['N_HEADS'], :, :], test_target)        # shuffle the test&cal sets and take the n_heads heads's outputs
    cal_scores = compute_scores(cal_output_new,config)   #calculate the nonconformity score
    test_scores = compute_scores(test_output_new,config)
    Dcells_scores,Dcells_target,Dre_cal_scores,Dre_cal_target = generate_Dcal_Dcells_sets(cal_scores, cal_target_new) #split  Dcal to Dcells and Dre-cal

    coverage, mean_set,prediction_sets_new_meth,_ = main_algo(Dcells_scores, Dcells_target,Dre_cal_scores, Dre_cal_target, test_scores,test_target_new, config['ALPHA'],config)
    print(f"\n{' Results ':#^50}\n")
    print(f"Coverage archived: {coverage}\n")
    print(f"Coverage wanted: {1-config['ALPHA']}\n")
    print(f"Mean-Predicted-Set-Size: {mean_set}\n")





if __name__=="__main__":
    config_name = 'multi_dim_cp_config'
    config = load_config(config_name)
    run(config)




