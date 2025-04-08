from dataclasses import dataclass, field
import numpy as np
from scipy.special import comb

@dataclass
class CoxRossRubinstein:
    s0: float
    u: float
    d: float
    r: float = 0.0
    q: float = field(init=False)

    def __post_init__(self):
        self.q = (self.r - self.d) / (self.u - self.d)

    def path_indep_price(self, maturity, payoff):
        N = np.arange(maturity+1)
        Q = comb(maturity, N) * self.q**N * (1-self.q)**N[::-1]
        S = self.s0 * (1+self.u)**N * (1+self.d)**N[::-1]
        return np.dot(Q, payoff(S)) / (1+self.r)**maturity
        
    def call_price(self, maturity, strike):
        payoff = lambda s: np.maximum(np.subtract.outer(s, strike), 0)
        return self.path_indep_price(maturity, payoff)
    
    def put_price(self, maturity, strike):
        payoff = lambda s: np.maximum(-np.subtract.outer(s, strike), 0)
        return self.path_indep_price(maturity, payoff)
      
    def path_dep_price(self, maturity, payoff):
        Omega = np.unpackbits(np.arange(2**maturity, dtype='<u8').view('u1').\
            reshape((-1, 8)), axis=1, count=maturity, bitorder='little').T
        Q = (self.q/(1-self.q))**np.sum(Omega, axis=0) * (1-self.q)**maturity
        xi = np.where(Omega, 1+self.u, 1+self.d)
        S = self.s0*np.vstack((np.ones(2**maturity), np.cumprod(xi, axis=0)))
        X = payoff(S)
        return np.dot(Q, X) / (1+self.r)**maturity
    
    def simulate(self, t, paths, rng=None):
        if rng is None:
            rng = np.random.default_rng()        
        xi = rng.choice(a=(1+self.u, 1+self.d), p=(self.q, 1-self.q), size=(t,paths))
        return self.s0*np.vstack([np.ones((1,paths)), xi.cumprod(axis=0)])
    
    def american_price(self, maturity, payoff):
        def X(t):
            N = np.arange(t+1)
            return payoff(self.s0 * (1+self.u)**N * (1+self.d)**(t-N))
        V = X(maturity)
        for t in range(maturity-1, -1, -1):
            V[:t+1] = np.maximum(X(t), (self.q*V[1:t+2] + (1-self.q)*V[:t+1])/(1+self.r))
        return V[0]
    
    def american_call_price(self, maturity, strike):
        return self.american_price(maturity, lambda s: np.maximum(s-strike, 0))
    
    def american_put_price(self, maturity, strike):
        return self.american_price(maturity, lambda s: np.maximum(strike-s, 0))

    
