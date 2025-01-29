from sklearn import linear_model
import numpy as np

from rpy2.robjects import r
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

MARS = importr('earth')

rpy2.robjects.numpy2ri.activate()

""" Basic functionalities for MARS spline regression and GLOBE MDL scores for splines. """

def REarth(X, Y, M=1):
	row, col = X.shape;
	rX = r.matrix(X, ncol=col, byrow=False);
	rY = r.matrix(Y, ncol=1, byrow=True);

	try:
		rearth = MARS.earth(x=rX, y=rY, degree=M);
	except:
		print("Singular fit encountered, retrying with Max Interactions=1");
		rearth = MARS.earth(x=rX, y=rY, degree=1);

	RSS_INDEX = 0;
	DIRS_INDEX = 5;
	CUTS_INDEX = 6;
	SELECTED_INDEX = 7;
	COEFF_INDEX = 11;

	no_of_terms = np.size(rearth[SELECTED_INDEX]);

	# print('-------')
	# first we extract the hinges that were finally selected by MARS
	working_index = np.array(rearth[SELECTED_INDEX].flatten(), dtype=int) - 1;
	# print("WI: ",working_index)
	# print("Orig: ",rearth[SELECTED_INDEX].flatten())

	# next we check if these selected hinges contain all the variables that were present in X
	dir_rows = rearth[DIRS_INDEX][working_index, :];
	dirs = np.sum(np.abs(dir_rows), axis=0);
	unused = (len(np.flatnonzero(dirs)) + 1) < X.shape[
		1];  # +1 is added to take into account the all 1's column, seems like MARS uses its own intercept term so our 1's col is set to zero always.
	# print('-------')
	# print("Dirs: ",dirs)
	# print("Unused: ",unused)

	# next we would like to know the number of terms in each hinge
	# we can do this by taking row sum of the selected Dirs
	interactions = [];
	for j in range(dir_rows.shape[0]):
		int_row = dir_rows[j, :];
		ints = np.sum(np.array(int_row != 0, dtype=int))
		interactions.append(ints);

	# print('-------')
	# print(interactions);
	# next we would like to record the coefficients
	# print('-------')
	coeffs = [];
	cut_rows = rearth[CUTS_INDEX][working_index, :];
	for j in range(cut_rows.shape[0]):
		c_row = cut_rows[j, :];
		c_index = np.flatnonzero(c_row);
		for ci in c_index:
			coeffs.append(c_row[ci]);
	# print("Coeff: ",coeffs)

	reg_coeffs = rearth[COEFF_INDEX].reshape((-1, 1));
	for j in range(reg_coeffs.shape[0]):
		coeffs.append(reg_coeffs[j, 0]);
	# print("Coeff: ",coeffs)

	sse = rearth[RSS_INDEX][0]
	# print("sse: ",sse)

	return sse, [coeffs], np.array([no_of_terms]), interactions;


def getStructuralDistances(G, H):
	return -1, -1, -1


class Slope:
	""" SLOPE, part of the GLOBE implementation (Mian et al. 2021).
	"""
	def __init__(self):
		self.inclusive_model=True;
	
	
	def FitSpline(self,source,target,M=2,temp=False):
		sse,coeff,hinge_count,interactions = REarth(source,target,M)
		score = self.model_score(np.copy(coeff[0])) 
		return sse,score,coeff,hinge_count,interactions;
	
	def FitModel(self,source,target,temp=False):
		reg = linear_model.LinearRegression(fit_intercept =False);
		reg.fit(source,target);
		coeff = reg.coef_;
		sse  = np.sum( (reg.predict(source) - target)**2 );
		score = self.model_score(np.copy(coeff[0]));
		return sse,score,coeff;

	
	def model_score(self,coeff):
		Nans = np.isnan(coeff);
		
		if any(Nans):
			print ('Warning: Found Nans in regression coefficients. Setting them to zero...')
		coeff[Nans]=0;
		sum =0;
		for c in coeff:
			if np.abs(c)>1e-12:
				c_abs =  np.abs(c);
				c_dummy = c_abs;
				precision = 1;
				
				while c_dummy<1000:
					c_dummy *=10;
					precision+=1;
				sum = sum + self.logN(c_dummy) + self.logN(precision) + 1
		return sum;
	
	def logN(self,z):
		z = np.ceil(z);
		
		if z < 1 :
			return 0;
		else :
			log_star = self.logg(z);
			sum = log_star;
			
			while log_star > 0:
				log_star = self.logg(log_star);
				sum = sum+log_star;
			
			return sum + self.logg(2.865064)
		
	def gaussian_score_emp_sse(self,sse, n,min_diff):
		var = sse / n
		sigma = np.sqrt(var)
		return self.gaussian_score_sse(sigma, sse, n,min_diff)

	def gaussian_score_sse (self,sigma, sse, n,resolution):
		sigmasq = sigma**2;
		if sse == 0.0 or sigmasq == 0.0:
			return np.array([0.0]);
		else:
			err = (sse / (2 * sigmasq * np.log(2))) + ((n/2) * self.logg(2 * np.pi * sigmasq)) - n * self.logg(resolution)
			return max(err,np.array([0]));
		
	def logg(self,x):
		if x == 0:
			return 0
		else:
			return np.log2(x)
