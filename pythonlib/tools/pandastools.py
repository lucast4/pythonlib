""" tools for use with pandas dataframes. also some stuff using python dicts and translating between that and dataframs"""
import pandas as pd



#############################vvvv OBSOLETE - USE aggregGeneral
def aggreg(df, group, values, aggmethod=["mean","std"]):
	"""
	get group means, for certain dimensions(values). 
	e.g., group = ["worker_model", "worker_dat"]
	e.g. values = ["score", "cond_is_same"]
	NOTE: will change name of balues filed, e.g. to score_mean.
	OBSOLETE - USE aggregGeneral]
	"""
	# this version did not deal with non-numeric stuff that would liek to preset
	# but was useful in taking both mean and std.
	df = df.groupby(group)[values].agg(aggmethod).reset_index()
	# df.columns = df.columns.to_flat_index()
	df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
	return df

def aggregMean(df, group, values, nonnumercols=[]):
	"""
	get group means, for certain dimensions(values). 
	e.g., group = ["worker_model", "worker_dat"]
	e.g. values = ["score", "cond_is_same"]
	e.g. nonnumercols=["sequence", "name"] i.e., will take the first item it encounters.
	[OBSOLETE - USE aggregGeneral]
	"""
	agg = {c:"mean" for c in df.columns if c in values }
	agg.update({c:"first" for c in df.columns if c in nonnumercols})
	print(agg)
	df = df.groupby(group).agg(agg).reset_index()
	# df.columns = df.columns.to_flat_index()
	# df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
	return df
########################### ^^^^ OBSOLETE - USE aggregGeneral


def aggregGeneral(df, group, values, nonnumercols=[], aggmethod=["mean"]):
	"""
	get group means, for certain dimensions(values). 
	e.g., group = ["worker_model", "worker_dat"]
	e.g. values = ["score", "cond_is_same"]
	e.g. nonnumercols=["sequence", "name"] i.e., will take the first item it encounters.
	"""
	agg = {c:aggmethod for c in df.columns if c in values }
	agg.update({c:"first" for c in df.columns if c in nonnumercols})
	print(agg)
	df = df.groupby(group).agg(agg).reset_index()
	# df.columns = df.columns.to_flat_index()

	if len(aggmethod)==1:
		# then reanme columns so same as how they came in:
		# e.g., dist instead of dist_mean. can't do if 
		# multiple aggmethods, since will then be ambiguos.
	# df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
		df.columns = [tup[0] for tup in df.columns.values]
	else:
		df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]

	return df

def df2dict(df):
	return df.to_dict("records")


def applyFunctionToAllRows(df, F, newcolname="newcol"):
    """F is applied to each row. is appended to original dataframe. F(x) must take in x, a row object"""
    return df.merge(df.apply(lambda x: F(x), axis=1).reset_index(), left_index=True, right_index=True).rename(columns={0:newcolname})


