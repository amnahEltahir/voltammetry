import pickle
import os
import numpy as np
from voltammetry import calcStepStats


def save_model(filename, cvFit, predictions, data, labels, outdir):
    """
    :param filename: name of original file containing data
    :param cvFit:
    :param outdir:
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        pass
    pkl_filename = os.path.join(outdir, filename + ".pkl")
    with open(pkl_filename, 'wb') as file:
        pickle.dump(cvFit, file)

    hdr_filename = os.path.join(outdir, filename + '.hdr')
    with open(hdr_filename, "w") as header_file:
        header_file.write(filename)
        header_file.write('\n' + "training sample size" + data.training.sampleSize)
        header_file.write('\n' + "testing sample size" + data.testing.sampleSize)
        attributes = ["labels", "prediction_RMSE", "prediction_SNR", "prediction_SNRE", "mean", "sd", "n", "sem",
                      "fullRMSE", "fullSNR", "fullSNRE"]
        for chemIx in range(len(labels.targetAnalyte)):
            header_file.write('\n' + labels.targetAnalyte[chemIx] + ':')
            stats = calcStepStats(chemIx, predictions, data.testing.labels)
            for attr in attributes:
                header_file.write('\n' + attr + ',')
                header_file.write(np.array2string(stats.__dict__.get(attr)))
