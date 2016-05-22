"""
Created 2016-05-20 by Janos Szalai-Gindl
"""
import numpy as np
from numpy.linalg import inv
from pymc.diagnostics import effective_n

class AutoCorrUtil:
    def autocorr(self, k, n, data, m_data, var_data, idx):
        cov_data = 0.0
        for t in range(0, n-k):
            cov_data += float(data[t][idx] - m_data)*float(data[t+k][idx] - m_data)
        cov_data/=float(n)
        return cov_data / var_data

    def autocorrfunc(self, until, n, data, idx):
        lst = []
        m_data = np.mean(data, axis = 0)[idx]
        var_data = np.var(data, axis = 0)[idx]
        for k in range(1,until):
            lst.append(self.autocorr(k, n, data, m_data, var_data, idx))
        return lst
    
    def effectiveSampleSize(self, until, n, data, idx):
        m_data = np.mean(data, axis = 0)[idx]
        var_data = np.var(data, axis = 0)[idx]
        autoCorrTime = 0
        for k in range(1,until):
            autoCorrTime += self.autocorr(k, n, data, m_data, var_data, idx)
        autoCorrTime = 1 + 2 * autoCorrTime
        return float(n)/float(autoCorrTime)

    # This is similar to the effective sample size method of the diagnostics of the pymc package:
    def effectiveSampleSizeMod(self, until, n, data, idx):
        m_data = np.mean(data, axis = 0)[idx]
        var_data = np.var(data, axis = 0)[idx]
        autoCorrTime = 0
        autoCorr_prev = self.autocorr(1, n, data, m_data, var_data, idx)
        for k in range(1,until):
            autoCorrTime += self.autocorr(k, n, data, m_data, var_data, idx)
            autoCorr1 = self.autocorr(k+1, n, data, m_data, var_data, idx)
            autoCorr2 = self.autocorr(k+2, n, data, m_data, var_data, idx)
            autoCorrSum = autoCorr1 + autoCorr2
            if((k%2==1)and(autoCorrSum < 0)):
                print k
                break
        autoCorrTime = 1 + 2 * autoCorrTime
        return float(n)/float(autoCorrTime)

    def effectiveSampleSizePymc(self, until, n, data, idx):
        traces = np.array([data[:,idx],data[:,idx]])
        return effective_n(traces)
		
    def finiteEffectiveSampleSize(self, until, n, data, idx):
        m_data = np.mean(data, axis = 0)[idx]
        var_data = np.var(data, axis = 0)[idx]
        finiteAutoCorrTime = 0
        for k in range(1,until):
            finiteAutoCorrTime += ((1-float(k)/float(n))*self.autocorr(k, n, data, m_data, var_data, idx))
        finiteAutoCorrTime = 1 + 2 * finiteAutoCorrTime
        return float(n)/float(finiteAutoCorrTime)