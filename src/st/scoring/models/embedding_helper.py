#import numpy as np
import numpy as np
import scipy as scipy


def cartesian(arrays, out=None):
		"""
		Generate a cartesian product of input arrays.

		Parameters
		----------
		arrays : list of array-like
				1-D arrays to form the cartesian product of.
		out : ndarray
				Array to place the cartesian product in.

		Returns
		-------
		out : ndarray
				2-D array of shape (M, len(arrays)) containing cartesian products
				formed of input arrays.

		Examples
		--------
		>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
		"""
		arrays = [np.asarray(x) for x in arrays]
		dtype = arrays[0].dtype

		n = np.prod([x.size for x in arrays])
		if out is None:
				out = np.zeros([n, len(arrays)], dtype=dtype)

		m = n / arrays[0].size
		m = int(m)
		out[:,0] = np.repeat(arrays[0], m)
		if arrays[1:]:
				cartesian(arrays[1:], out=out[0:m,1:])
				for j in range(1, arrays[0].size):
						out[j*m:(j+1)*m,1:] = out[0:m,1:]
		return out

def interval(n,d,L_infinity_ball = 1):
	arrays = [np.linspace(-L_infinity_ball,L_infinity_ball,n).reshape(n,1) for i in range(d)]
	xtest = cartesian(arrays)
	return xtest


def logsumexp(a, axis=None, b=None):
	a = np.asarray(a)
	if axis is None:
		a = a.ravel()
	else:
		a = np.rollaxis(a, axis)
	a_max = a.max(axis=0)
	if b is not None:
		b = np.asarray(b)
		if axis is None:
			b = b.ravel()
		else:
			b = np.rollaxis(b, axis)
		out = np.log(np.sum(b * np.exp(a - a_max), axis=0))
	else:
		out = np.log(np.sum(np.exp(a - a_max), axis=0))
	out += a_max
	return out

class MyBounds(object):
	def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
		self.xmax = np.array(xmax)
		self.xmin = np.array(xmin)

	def __call__(self, **kwargs):
		x = kwargs["x_new"]
		tmax = bool(np.all(x <= self.xmax))
		tmin = bool(np.all(x >= self.xmin))
		return tmax and tmin


def next_prime():
		def is_prime(num):
				"Checks if num is a prime value"
				for i in range(2,int(num**0.5)+1):
						if(num % i)==0: return False
				return True

		prime = 3
		while(1):
				if is_prime(prime):
						yield prime
				prime += 2

def vdc(n, base=2):
		vdc, denom = 0, 1
		while n:
				denom *= base
				n, remainder = divmod(n, base)
				vdc += remainder/float(denom)
		return vdc



def full_group(d):
	g = []
	for i in range(d):
		g.append([i])
	return g

def pair_groups(d):
	g = []
	for i in range(d-1):
		g.append([i,i+1])
	return g


class results:
	def __init__(self):
		self.x = 0


def  proj(x,bounds):
	y = np.zeros(shape = x.shape)
	for ind,elem in enumerate(x):
		if elem > bounds[ind][1]:
			y[ind] = bounds[ind][1]

		elif elem < bounds[ind][0]:
			y[ind] = bounds[ind][0]

		else:
			y[ind] = elem
	return y


def sample_bounded(bounds):
	d = len(bounds)
	x = np.zeros(shape = (d))
	for i in range(d):
		x[i] = np.uniform(bounds[i][0],bounds[i][1])
	return x


def lambda_coordinate(fun,x0,index,x):
	x0[index] = x
	r = fun(x0)
	return r


def halton_sequence(size, dim):
		seq = []
		primeGen = next_prime()
		next(primeGen)
		for d in range(dim):
				base = next(primeGen)
				seq.append([vdc(i, base) for i in range(size)])
		return seq


def sample_qmc_halton_normal(size = (1,1)):
	Z = np.array(halton_sequence(size[0],size[1])).T
	Z[0,:] += 10e-5
	from scipy.stats import norm
	Z = norm.ppf(Z)
	return Z

def sample_qmc_halton(sampler, size = (1,1)):
	Z = np.array(halton_sequence(size[0],size[1])).T
	Z[0,:] += 10e-5
	Z = sampler(Z)
	return Z

def rejection_sampling(pdf, size = (1,1)):
	"""
	Implements rejection sampling

	:param pdf:
	:param size:
	:return:
	"""
	n = size[0]
	d = size[1]
	from scipy.stats import norm
	output = np.zeros(shape =size)
	i = 0
	while i < n:
		Z = np.random.normal (size = (1,d))
		u = np.random.uniform()
		if pdf(Z) < u:
			output[i,:] = Z
			i=i+1

	return output


def sample_custom(inverse_cumulative_distribution,size = (1,1)):
	U = np.random.uniform(0,1,size = size)
	F = np.vectorize(inverse_cumulative_distribution)
	Z = F(U)
	return Z


if __name__=="__main__":
	#print (sample_qmc_halton(size = (100,2) ))

	fun = lambda x: np.sin(np.sum(3*x**2))
	x = np.array([1,1]).reshape(1,2)
	complex_step_derivative(fun,1e-8,x)
