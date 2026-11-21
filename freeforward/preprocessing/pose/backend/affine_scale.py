"""Regularized shared additive-depth offsets and positive window scales.

All inputs are frozen RGB predictions. The optional empirical scale band is a
stability assumption, not a statistical confidence interval. Standard robust
regression and Schur elimination are not claimed as new numerical methods.
"""
import numpy as np


def huber(z):
 a=abs(z);return np.where(a<=1.5,.5*a*a,1.5*(a-.75))


def fit(n_windows,observations,bounded=True,shared_offset=True,tau=.05,max_iter=100,support_bands=None):
 if not observations or tau<=0:raise ValueError('Empty or invalid affine problem')
 images=sorted({o['image'] for o in observations});lookup={s:i for i,s in enumerate(images)};seen=set();counts=np.zeros(n_windows);ratios=[[] for _ in range(n_windows)];q={}
 for o in observations:
  w=o['window'];key=(w,o['image'])
  if key in seen:raise ValueError('Duplicate window/image observation')
  seen.add(key)
  d,m=np.asarray(o['relative'],float),np.asarray(o['metric'],float)
  if not (0<=w<n_windows) or d.ndim!=1 or d.shape!=m.shape or len(d)<3 or not np.isfinite(d).all() or not np.isfinite(m).all() or np.any(d<=0) or np.any(m<=0):raise ValueError('Invalid depth observation')
  counts[w]+=1;ratios[w].extend((m/d).tolist())
  qi=float(o['metric_reference'])
  if qi<=0 or not np.isfinite(qi):raise ValueError('Invalid metric reference')
  if o['image'] in q and q[o['image']]!=qi:raise ValueError('A physical image must share its metric reference')
  q[o['image']]=qi
 if np.any(counts==0):raise ValueError('Window has no metric-depth anchor')
 initial=np.array([np.median(v) for v in ratios]);bands=np.array([np.quantile(v,[.4,.6]) for v in ratios]) if support_bands is None else np.asarray(support_bands,float)
 if bands.shape!=(n_windows,2) or not np.isfinite(bands).all() or np.any(bands<=0) or np.any(bands[:,0]>bands[:,1]):raise ValueError('Invalid empirical scale band')
 lo=bands[:,0]/initial if bounded else np.full(n_windows,1e-6);hi=bands[:,1]/initial if bounded else np.full(n_windows,np.inf)
 ww=[];ii=[];aa=[];cc=[];yy=[];mm=[]
 for o in observations:
  w=o['window'];d,m=np.asarray(o['relative'],float),np.asarray(o['metric'],float);qi=q[o['image']];sig=.05*np.maximum(m/qi,.05)
  ww.extend([w]*len(d));ii.extend([lookup[o['image']]]*len(d));aa.extend(initial[w]*d/qi/sig);cc.extend(1/sig);yy.extend(m/qi/sig);mm.extend(np.full(len(d),1/(counts[w]*len(d))))
 w,im=np.array(ww,int),np.array(ii,int);a,c,y,mass=map(np.array,[aa,cc,yy,mm]);ni=len(images)
 x=np.clip(np.ones(n_windows),lo,hi);beta=np.zeros(ni)
 def objective(t,b):return float(mass@huber(a*t[w]+c*b[im]-y)+(.5*np.sum((b/tau)**2) if shared_offset else 0.))
 history=[objective(x,beta)];converged=False
 for iteration in range(max_iter):
  residual=abs(a*x[w]+c*beta[im]-y);p=mass*np.minimum(1.,1.5/np.maximum(residual,1e-12))
  dx=np.bincount(w,weights=p*a*a,minlength=n_windows);rx=np.bincount(w,weights=p*a*y,minlength=n_windows)
  if shared_offset:
   db=np.bincount(im,weights=p*c*c,minlength=ni)+1/tau**2;rb=np.bincount(im,weights=p*c*y,minlength=ni)
   cross=np.bincount(w*ni+im,weights=p*a*c,minlength=n_windows*ni).reshape(n_windows,ni)
   H=np.diag(dx)-(cross/db)@cross.T;rhs=rx-cross@(rb/db);nx=x.copy()
   for sweep in range(1000):
    before=nx.copy()
    for k in range(n_windows):nx[k]=np.clip((rhs[k]-H[k]@nx+H[k,k]*nx[k])/H[k,k],lo[k],hi[k])
    if np.max(abs(nx-before))<1e-10:break
   nb=(rb-cross.T@nx)/db
  else:nx=np.clip(rx/dx,lo,hi);nb=np.zeros(ni)
  value=objective(nx,nb)
  if value>history[-1]+1e-7:raise RuntimeError('Non-monotone affine solve')
  change=max(np.max(abs(nx-x)),np.max(abs(nb-beta)));x,beta=nx,nb;history.append(value)
  if change<1e-8:converged=True;break
 if not bounded and np.any(x<=1.01e-6):raise ValueError('Unbounded solution reaches positive-scale boundary')
 return dict(scales=(initial*x).tolist(),initial_ratio_medians=initial.tolist(),empirical_bands=bands.tolist(),
             offsets_m={s:float(beta[i]*q[s]) for s,i in lookup.items()},normalized_offsets=dict(zip(images,beta.tolist())),
             tau=tau,bounded=bounded,shared_offset=shared_offset,at_bound=np.flatnonzero(np.isclose(x,lo,atol=1e-7)|np.isclose(x,hi,atol=1e-7)).tolist(),
             iterations=iteration+1,converged=converged,objective_history=history,metric_images=ni,observations=len(observations),points=len(w),GT_used=False)
