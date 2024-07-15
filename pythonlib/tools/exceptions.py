class NotEnoughDataException(Exception):
	"""useful to skip some anakysis in context of a batch analys"""
	pass
		

class DataMisalignError(Exception):
	""" If there are weird misalingments in data
	e.g, mapping neural to beh
	"""
	pass

class RuleDoesntExist(Exception):
	"""i.e. this is not a grammar rule dataset.."""
	pass
