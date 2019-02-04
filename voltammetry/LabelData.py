import pandas as pd
import numpy as np


class Mulabels:
    """
    Concentration labels according to their order in the sequence
    """

    def __init__(self, data_dir, label_file_name):
        """Object read from CSV containing data frame of sequence labels"""
        self.data = pd.read_csv(''.join((data_dir, '/', label_file_name)))
        self.chems = list(self.data.columns.intersection(['DA', '5HT', '5HIAA', 'NE', 'pH']))
        self.labels = self.data[self.chems]
        target_analyte = []
        chem_ix = []
        for col in list(self.chems):
            n_unique = len(np.unique(self.labels[col]))
            if n_unique > 1:
                target_analyte.append(col)
                chem_ix.append(self.labels.columns.get_loc(col))
        self.targetAnalyte = target_analyte
        self.chemIx = chem_ix
