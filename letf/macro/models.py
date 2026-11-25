"""Transparent competing joint monthly path models, fit on training only."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from letf.accuracy import UnsupportedModelError


def _state(row,columns):
    # Fixed observable stress definitions; no ex-post recession labels.
    pi=columns.index('inflation_log');g=columns.index('production_log')
    return int(12*row[pi]>.04)+2*int(row[g]<0)


@dataclass
class MacroModel:
    kind:str
    columns:list
    train_values:np.ndarray
    parameters:dict

    @classmethod
    def fit(cls,kind,training,protocol):
        x=training.to_numpy(float)
        if kind not in ('joint_block','regime_block','conditional_var_t'):
            raise UnsupportedModelError('Unregistered model type')
        if not np.isfinite(x).all() or not len(x):raise ValueError('Finite training values required')
        minimum=protocol['minimum_conditional_months'] if kind=='conditional_var_t' else protocol['minimum_bootstrap_months']
        if len(x)<minimum:raise UnsupportedModelError('Insufficient training observations for model')
        parameters={'block_months':protocol['block_months'],'training_end':str(training.index[-1].date()),
            'training_start':str(training.index[0].date()),'training_months':len(x)}
        if kind=='conditional_var_t':
            mean=x.mean(0);scale=x.std(0,ddof=1)
            if (scale<=1e-12).any():raise UnsupportedModelError('Constant model variable lacks identified conditional dynamics')
            z=(x-mean)/scale;design=np.column_stack([np.ones(len(x)-1),z[:-1]])
            penalty=np.diag([0.,*([protocol['ridge_penalty']]*x.shape[1])])
            coefficients=np.linalg.solve(design.T@design+penalty,design.T@z[1:])
            a=coefficients[1:];radius=float(max(abs(np.linalg.eigvals(a))))
            if radius>protocol['max_var_spectral_radius']:
                coefficients[1:]*=protocol['max_var_spectral_radius']/radius
            residual=z[1:]-design@coefficients
            covariance=np.cov(residual,rowvar=False)
            shrink=protocol['covariance_diagonal_shrinkage']
            covariance=(1-shrink)*covariance+shrink*np.diag(np.diag(covariance))
            if np.linalg.eigvalsh(covariance).min()<=0:raise UnsupportedModelError('Unidentified joint covariance')
            parameters.update(mean=mean.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),
                covariance=covariance.tolist(),df=protocol['student_df'],spectral_radius_before=radius,
                spectral_radius_after=float(max(abs(np.linalg.eigvals(coefficients[1:])))))
        return cls(kind,list(training.columns),x.copy(),parameters)

    def simulate(self,months,paths,seed,*,constraints=None,vol_multiplier=1.):
        if type(months)is not int or months<1 or type(paths)is not int or paths<1:raise ValueError('Positive path/horizon counts required')
        if not np.isfinite(vol_multiplier) or vol_multiplier<=0:raise ValueError('Positive volatility multiplier required')
        rng=np.random.default_rng(seed);d=len(self.columns);out=np.empty((paths,months,d))
        constraints=constraints or {}
        if set(constraints)-set(self.columns):raise UnsupportedModelError('Unknown macro constraint')
        if any(not np.isfinite(v) for v in constraints.values()):raise ValueError('Nonfinite scenario constraint')
        if self.kind in ('joint_block','regime_block'):
            if constraints or vol_multiplier!=1:
                raise UnsupportedModelError('Bootstrap shocks need an explicit conditional model; independent asset overrides are forbidden')
            block=self.parameters['block_months'];x=self.train_values
            starts=np.arange(len(x)-block+1)
            states=np.array([_state(r,self.columns) for r in x])
            current=np.full(paths,_state(x[-1],self.columns))
            for t in range(0,months,block):
                length=min(block,months-t)
                for p in range(paths):
                    eligible=starts if self.kind=='joint_block' else starts[states[starts]==current[p]]
                    if not len(eligible):eligible=starts  # documented finite-history fallback
                    start=int(rng.choice(eligible))
                    out[p,t:t+length]=x[start:start+length]
                    current[p]=states[start+length-1]
            return out
        par=self.parameters;mean=np.asarray(par['mean']);scale=np.asarray(par['scale'])
        coefficients=np.asarray(par['coefficients']);covariance=np.asarray(par['covariance'])*vol_multiplier**2
        nu=par['df'];z=np.broadcast_to((self.train_values[-1]-mean)/scale,(paths,d)).copy()
        fixed=np.array([self.columns.index(c) for c in constraints],dtype=int)
        free=np.array([i for i in range(d) if i not in fixed],dtype=int)
        fixed_target=np.array([(constraints[self.columns[i]]-mean[i])/scale[i] for i in fixed])
        # Student-t scale matrix; covariance = scale_matrix*nu/(nu-2).
        s=covariance*(nu-2)/nu
        for t in range(months):
            mu=coefficients[0]+z@coefficients[1:]
            if len(fixed):
                cc=s[np.ix_(fixed,fixed)];fc=s[np.ix_(free,fixed)]
                residual=fixed_target-mu[:,fixed]
                solved=np.linalg.solve(cc,residual.T).T
                conditional_mean=mu[:,free]+solved@fc.T
                conditional_scale=s[np.ix_(free,free)]-fc@np.linalg.solve(cc,fc.T)
                delta=np.einsum('ij,ij->i',residual,solved)
                df=nu+len(fixed)
                innovation=rng.standard_normal((paths,len(free)))@np.linalg.cholesky(conditional_scale).T if len(free) else np.empty((paths,0))
                innovation*=np.sqrt((nu+delta)/df)[:,None]/np.sqrt(rng.chisquare(df,paths)/df)[:,None]
                z[:,free]=conditional_mean+innovation;z[:,fixed]=fixed_target
            else:
                innovation=rng.standard_normal((paths,d))@np.linalg.cholesky(s).T
                innovation/=np.sqrt(rng.chisquare(nu,paths)/nu)[:,None]
                z=mu+innovation
            out[:,t]=mean+z*scale
        if not np.isfinite(out).all():raise ValueError('Nonfinite model path; no failed paths are dropped')
        return out
