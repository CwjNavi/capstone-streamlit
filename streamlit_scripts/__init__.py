import numpy as np
from typing import Literal
from scipy.stats import norm

class BlackScholes:
    def init(self):
        pass
    
    def calculate_premium(self, options_type:Literal["put", "call"], St:float, X:float, rf:float, q:float, vol:float, T:float):
        d1 = self.d1(St=St, X=X, rf=rf, q=q, vol=vol, T=T)
        d2 = self.d2(d1=d1, vol=vol, T=T)
        
        if options_type == "call":
            return np.exp(-q*T)*St*norm.cdf(d1) - np.exp(-rf*T)*X*norm.cdf(d2)
        elif options_type == "put":
            return np.exp(-rf*T)*X*norm.cdf(-d2) - np.exp(-q*T)*St*norm.cdf(-d1)
    
    def d1(self, St:float, X:float, rf:float, q:float, vol:float, T:float):
        return (np.log(St/X) + (rf-q+0.5*vol**2)*T)/(vol*np.sqrt(T))
    
    def d2(self, d1:float, vol:float, T:float):
        return d1 - vol*np.sqrt(T)