"""
Created 2016-05-20 by Janos Szalai-Gindl
"""
import numpy as np

class AutoCorrUtil:
    def autocorr(self, k, n, data, idx):
        m_data = np.mean(data, axis = 0)[idx]
        var_data = np.var(data[:n-k], axis = 0)[idx]
        cov_data = 0.0
        for t in range(0, n-k):
            cov_data += float(data[t][idx] - m_data)*float(data[t+k][idx] - m_data)
        cov_data/=float(n)
        return cov_data / var_data

    def autocorrfunc(self, until, n, data, idx):
        lst = []
        for k in range(0,until):
            lst.append(self.autocorr(k, n, data, idx))
        return lst
    
    def effectiveSampleSize(self, until, n, data, idx):
        autoCorrTime = 0
        for k in range(0,until):
            autoCorrTime += self.autocorr(k, n, data, idx)
        autoCorrTime = 1 + 2 * autoCorrTime
        return float(n)/float(autoCorrTime)