import pickle
import os
from collections import Iterable
import numpy as np
from voltammetry import calcStepStats
import scipy.io as sio


def save_model(prefix, cvFit, predictions, data, labels, out_dir="OUT"):
    """
    :param prefix: str - prefix of hdr, mat and pkl files generated for model
    :param cvFit: fit - cross validation fit from cvglmnet
    :param predictions: array of predictions generated from cvFit
    :param data: Data - object containing with training/testing split
    :param labels: MuLabels - object containing concentration label info
    :param out_dir: str - location to output model info (default "OUT/")
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        pass
    pkl_filename = os.path.join(out_dir, prefix + ".pkl")
    with open(pkl_filename, 'wb') as file:
        pickle.dump(cvFit, file)
    mat_filename = os.path.join(out_dir, prefix + ".mat")
    sio.savemat(mat_filename, mdict={'cvFit': cvFit})
    hdr_filename = os.path.join(out_dir, prefix + '.hdr')
    with open(hdr_filename, "w") as header_file:
        header_file.write(prefix)
        header_file.write('\n' + "training sample size: " + str(data.training.sampleSize))
        header_file.write('\n' + "testing sample size: " + str(data.testing.sampleSize))
        attributes = ["labels", "prediction_RMSE", "prediction_SNR", "prediction_SNRE", "mean", "sd", "n", "sem",
                      "fullRMSE", "fullSNR", "fullSNRE"]
        for chemIx in range(len(labels.targetAnalyte)):
            header_file.write('\n---\n' + labels.targetAnalyte[chemIx] + ':')
            stats = calcStepStats(chemIx, predictions, data.testing.labels)

            for attr in attributes:
                if isinstance(stats.__dict__.get(attr), Iterable):
                    attr_str = ''
                    for mem in stats.__dict__.get(attr):
                        attr_str += str(mem) + '\t'

                else:
                    attr_str = np.array2string(stats.__dict__.get(attr))

                header_file.write('\n' + attr + '\t')
                header_file.write(attr_str)
