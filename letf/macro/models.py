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
        if not len(x) or np.isinf(x).any():raise ValueError('Finite training values required')
        incomplete=not np.isfinite(x).all()
        if incomplete and not protocol.get('allow_incomplete_training',False):raise ValueError('Finite training values required')
        complete=np.isfinite(x).all(1)
        minimum=protocol['minimum_conditional_months'] if kind=='conditional_var_t' else protocol['minimum_bootstrap_months']
        if complete.sum()<minimum or not complete[-1]:raise UnsupportedModelError('Insufficient training observations for model')
        parameters={'block_months':protocol['block_months'],'training_end':str(training.index[-1].date()),
            'training_start':str(training.index[0].date()),'training_months':len(x)}
        parameters.update(incomplete_training_months=int((~complete).sum()),
            estimation_scope='Observed adjacent complete transitions / complete blocks only; no filled observations')
        if kind=='conditional_var_t':
            mean=x[complete].mean(0);scale=x[complete].std(0,ddof=1)
            if (scale<=1e-12).any():raise UnsupportedModelError('Constant model variable lacks identified conditional dynamics')
            z=(x-mean)/scale;valid=complete[:-1]&complete[1:]
            design=np.column_stack([np.ones(len(x)-1),z[:-1]])[valid];target=z[1:][valid]
            penalty=np.diag([0.,*([protocol['ridge_penalty']]*x.shape[1])])
            coefficients=np.linalg.solve(design.T@design+penalty,design.T@target)
            a=coefficients[1:];radius=float(max(abs(np.linalg.eigvals(a))))
            if radius>protocol['max_var_spectral_radius']:
                coefficients[1:]*=protocol['max_var_spectral_radius']/radius
            residual=target-design@coefficients
            covariance=np.cov(residual,rowvar=False)
            shrink=protocol['covariance_diagonal_shrinkage']
            covariance=(1-shrink)*covariance+shrink*np.diag(np.diag(covariance))
            if np.linalg.eigvalsh(covariance).min()<=0:raise UnsupportedModelError('Unidentified joint covariance')
            parameters.update(observed_adjacent_transitions=int(valid.sum()),mean=mean.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),
                covariance=covariance.tolist(),df=protocol['student_df'],spectral_radius_before=radius,
                spectral_radius_after=float(max(abs(np.linalg.eigvals(coefficients[1:])))))
        return cls(kind,list(training.columns),x.copy(),parameters)

    def innovation_covariance(self,*,vol_multiplier=1.,correlation_overrides=None):
        if self.kind!='conditional_var_t':raise UnsupportedModelError('No parametric innovation covariance for bootstrap')
        covariance=np.asarray(self.parameters['covariance'])*vol_multiplier**2
        scale=np.asarray(self.parameters['scale'])
        covariance=covariance*scale[:,None]*scale[None,:]
        for (left,right),rho in (correlation_overrides or {}).items():
            if left not in self.columns or right not in self.columns or not np.isfinite(rho) or abs(rho)>1:
                raise ValueError('Invalid correlation intervention')
            i,j=self.columns.index(left),self.columns.index(right)
            if i==j:raise ValueError('Cannot intervene on diagonal correlation')
            covariance[i,j]=covariance[j,i]=rho*np.sqrt(covariance[i,i]*covariance[j,j])
        if np.linalg.eigvalsh(covariance).min()<=0:
            raise UnsupportedModelError('Requested correlation intervention is not positive definite with the remaining joint system')
        return covariance

    def simulate(self,months,paths,seed,*,constraints=None,vol_multiplier=1.,
                 intercept_shifts=None,correlation_overrides=None,constraint_schedule=None,initial_state=None):
        if type(months)is not int or months<1 or type(paths)is not int or paths<1:raise ValueError('Positive path/horizon counts required')
        if not np.isfinite(vol_multiplier) or vol_multiplier<=0:raise ValueError('Positive volatility multiplier required')
        rng=np.random.default_rng(seed);d=len(self.columns);out=np.empty((paths,months,d))
        constraints=constraints or {}
        intercept_shifts=intercept_shifts or {};constraint_schedule=constraint_schedule or {}
        if set(intercept_shifts)-set(self.columns):raise UnsupportedModelError('Unknown drift intervention')
        if any(not np.isfinite(v) for v in intercept_shifts.values()):raise ValueError('Nonfinite drift intervention')
        if set(constraints)-set(self.columns):raise UnsupportedModelError('Unknown macro constraint')
        if any(not np.isfinite(v) for v in constraints.values()):raise ValueError('Nonfinite scenario constraint')
        if self.kind in ('joint_block','regime_block'):
            if constraints or vol_multiplier!=1 or intercept_shifts or correlation_overrides or constraint_schedule:
                raise UnsupportedModelError('Bootstrap shocks need an explicit conditional model; independent asset overrides are forbidden')
            block=self.parameters['block_months'];x=self.train_values
            starts=np.arange(len(x)-block+1)
            starts=starts[np.array([np.isfinite(x[i:i+block]).all() for i in starts])]
            if not len(starts):raise UnsupportedModelError('No complete historical blocks remain')
            states=np.array([_state(r,self.columns) for r in x])
            initial=x[-1] if initial_state is None else np.asarray(initial_state,float)
            current=np.array([_state(row,self.columns) for row in np.broadcast_to(initial,(paths,d))])
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
        coefficients=np.asarray(par['coefficients'])
        if correlation_overrides:
            covariance=self.innovation_covariance(vol_multiplier=vol_multiplier,correlation_overrides=correlation_overrides)/scale[:,None]/scale[None,:]
        else:covariance=np.asarray(par['covariance'])*vol_multiplier**2
        state=self.train_values[-1] if initial_state is None else np.asarray(initial_state,float)
        if not np.isfinite(state).all():raise ValueError('Nonfinite initial information state')
        nu=par['df'];z=np.broadcast_to((state-mean)/scale,(paths,d)).copy()
        fixed=np.array([self.columns.index(c) for c in constraints],dtype=int)
        free=np.array([i for i in range(d) if i not in fixed],dtype=int)
        fixed_target=np.array([(constraints[self.columns[i]]-mean[i])/scale[i] for i in fixed])
        # Student-t scale matrix; covariance = scale_matrix*nu/(nu-2).
        s=covariance*(nu-2)/nu
        for t in range(months):
            mu=coefficients[0]+z@coefficients[1:]
            for name,change in intercept_shifts.items():mu[:,self.columns.index(name)]+=change/scale[self.columns.index(name)]
            if constraint_schedule:
                current=dict(constraints,**constraint_schedule.get(t,{}))
                if set(current)-set(self.columns) or any(not np.isfinite(v) for v in current.values()):
                    raise ValueError('Invalid dated conditional intervention')
                fixed=np.array([self.columns.index(c) for c in current],dtype=int)
                free=np.array([i for i in range(d) if i not in fixed],dtype=int)
                fixed_target=np.array([(current[self.columns[i]]-mean[i])/scale[i] for i in fixed])
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
