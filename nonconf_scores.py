import numpy as np

def val_score_phase(cal_true_scores,test_scores,test_target,alpha):
    alpha_hat = np.ceil((cal_true_scores.shape[0] + 1) * (1 - alpha)) / cal_true_scores.shape[0]
    qhat = np.quantile(cal_true_scores, min(alpha_hat,1.), method='higher')
    prediction_sets = test_scores <= qhat
    zero_rows_idx=prediction_sets.sum(axis=1)==0
    if zero_rows_idx.any():
        prediction_sets[zero_rows_idx,:]=test_scores[zero_rows_idx, :] == test_scores[zero_rows_idx, :].min(axis=1)[:, np.newaxis]
    coverage, mean_set = prediction_sets[range(len(prediction_sets)), test_target.astype(
                                                       int)].sum() / len(
        prediction_sets), prediction_sets.sum(axis=1).mean()
    return prediction_sets,coverage,mean_set,qhat

def conf_score_SAPS(smx,phase,params=[0.3]):
    gamma=params[0]
    if phase=='SCORES':
        val_pi = smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(smx, val_pi, axis=1)
        rank=(np.tile(np.arange(val_srt.shape[1])-1,(val_srt.shape[0],1))+np.random.rand(val_srt.shape[0]).reshape(-1,1))*gamma
        rank[:, 0] = val_srt[:, 0] * np.random.rand(val_srt.shape[0])
        rank[:,1:]=rank[:,1:]+val_srt[:, 0,np.newaxis]
        return [np.take_along_axis(rank, val_pi.argsort(axis=1), axis=1)]


def conf_score_RAPS(smx,phase,params=[5,0.05],rand=True):
    k_reg,lam_reg=params[0],params[1]
    reg_vec = np.array(k_reg * [0, ] + (smx.shape[1] - k_reg) * [lam_reg, ])[None, :]

    if phase == 'SCORES':
        n_val = smx.shape[0]
        val_pi = smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(smx, val_pi, axis=1)
        val_srt_reg = val_srt + reg_vec
        val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
        indicators = (val_srt_reg_cumsum - np.random.rand(n_val, 1) * val_srt_reg) if rand else val_srt_reg_cumsum - val_srt_reg
        return [np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)]

def conf_score_NAIVE(smx,phase):
    # if phase=='CAL':
    #      return [1-smx[np.arange(n), cal_labels]]
    # elif phase=='VAL':
    #     idx=np.argmax(smx,axis=1)
    #     res=smx > (1 - qhat[0])
    #     if No_NAN_PREDS:
    #         res[np.arange(len(res)),idx]=True
    #     return res
    if phase=='SCORES':
       return [1 - smx]


def conf_score_APS(smx,phase):
    # if phase=='CAL':
    #     cal_pi = smx.argsort(1)[:, ::-1]
    #     cal_srt = np.take_along_axis(smx, cal_pi, axis=1)
    #     cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    #
    #     return [cal_srt.cumsum(axis=1)[np.arange(n), cal_L] - (np.random.rand(n) * cal_srt[np.arange(n), cal_L])]
    # elif phase=='VAL':
    #     val_pi = smx.argsort(1)[:, ::-1]
    #     val_srt = np.take_along_axis(smx, val_pi, axis=1)
    #     indicators = (val_srt.cumsum(axis=1) - (np.random.rand(smx.shape[0],1) * val_srt)) <= qhat[0]
    #     if No_NAN_PREDS:
    #         indicators[:,0]=True
    #     return np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
    if phase=='SCORES':
        val_pi = smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(smx, val_pi, axis=1)
        indicators = (val_srt.cumsum(axis=1) - (np.random.rand(smx.shape[0], 1) * val_srt))
        return [np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)]