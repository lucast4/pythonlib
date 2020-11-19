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

############3333 SCRATCH NOTES

# Apoply function to group-wise:
# Note; I was unsure whether the index pulled out is the global index or within the group.
# I think it is glocal.
# F = lambda x: x.iloc[x["distance"].idxmin()]
# F = lambda x: SF.iloc[x["distance"].idxmin()]
# # F = lambda x: SF.iloc[1]
# SF.groupby(["task", "epoch"]).apply(F)

######### take rows based on max/min of for values of some column:
# SF.loc[SF.groupby(["task", "epoch"])["distance"].idxmax()]


# pivoting (to take multiple rows and make them colums)
# Y = SFagg.pivot(index="task", columns="epoch", values="distance_median")


def filterGroupsSoNoGapsInData(df, group, colname, values_to_check):
    """ filter df so that each group has at
    least one item for each of the desired values
    for the column of interest. useful for removing
    data which don't have data for all days, for eg...
    -- e.g., this only keeps tasks that have data for
    both epochs:
	    values_to_check = [1,2]
		colname = "epoch"
		group = "unique_task_name"
    """
    def F(x):
        """ True if has data for all values"""
        checks  = []
        for v in values_to_check:
            checks.append(v in x[colname].values)
        return all(checks)
    return df.groupby(group).filter(F)


def getCount(df, group, colname):
	""" return df grouped by group, and with one count value
	for each level in group, the name of that will be colname
	- colname must be a valid column from df
	"""

	return df.groupby(group)[colname].count().reset_index()