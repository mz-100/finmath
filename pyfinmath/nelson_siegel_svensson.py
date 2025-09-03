from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import datetime

@dataclass
class NelsonSiegelSvensson:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    @classmethod
    def from_feds(cls, date=None, filename=None):
        if date is None:
            date = datetime.date.today()
        data = pd.read_csv(filename, skiprows=9, index_col='Date', parse_dates=True)
        nss_params = data.iloc[data.index.get_indexer([date], method='nearest')[0]]
        return cls(nss_params.BETA0, nss_params.BETA1, nss_params.BETA2, nss_params.BETA3, nss_params.TAU1, nss_params.TAU2)

    def yield_to_maturity(self, t):
        return (self.beta0 + self.beta1*(1-np.exp(-t/self.tau1))/(t/self.tau1) + 
               self.beta2*((1-np.exp(-t/self.tau1))/(t/self.tau1) - np.exp(-t/self.tau1)) + 
               self.beta3*((1-np.exp(-t/self.tau2))/(t/self.tau2) - np.exp(-t/self.tau2))) / 100
    
    def __call__(self, t):
        return self.yield_to_maturity(t)

    def discount_factor(self, t):
        return np.exp(-self(t)*t)
    