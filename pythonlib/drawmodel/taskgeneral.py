""" General TaskClass
(made during drawnn but should be extendable for other purposes
"""
from . import primitives as Prim
import numpy as np


class TaskClass(object):
	def __init__(self, program=None, shapes=None):
		"""
		- program is list of subprograms, each suprogram is list of 
		lines, where each line evaluates and takes in the accumulation of
		previous lines within the subprogram. e..g,:
			program = []
			program.append(
			    [["line"],
			    ["repeat", {"p":"$", "N":3, "x":0.5}]])
			program.append(
			    [["circle", [None, None, 0.5, 0.5, None, None]],
			     ["line"],
			     ["repeat", {"y":0.5}]]
			)
		Note: each line is like [<function>, <params>], where params can be
		--- dict that will be passed in as kwargs. Note that "$" means take in
		accumulation of previousl lines in this subprogram as this argument. If 
		dont pass in $, then will assume to pass in previous lines into first
		position argument (e..g,, in repeat in 2nd suprogram).
		--- list, this must not miss any arguemnts, but can pass None if to use
		default. Can choose to pass in the $, or not, if not then uses as first arg.

		- shapes, is list of shape (single stroke) primitives. this will be automatcaly
		extracted from program. these are still symbolic. 
		--- can pass in shapes, in which case program will be None. I did this since 
		havent figured out best way to evaluate program.

		- strokes, is like shapes, but in numbers. This is computed automatically.
		"""

		a = program is not None
		b = shapes is not None
		assert a!=b, "provide either program or shapes"

		self.Program = program

		if program is not None:
			self.Shapes = self.program2shapes(self.Program)
		else:
			self.Shapes = shapes
		
		self.Strokes = self.shapes2strokes(self.Shapes)

		self.Points = np.stack([ss for s in self.Strokes for ss in s], axis=0) # flatten Strokes


	def program2shapes(self, program):
		assert False, "not done, see __init__ for what need to do."
		shapes =[]
		for subprog in program:

			funcstring = evaluateProg(subprog) 
			subshapes = evaluateString(funcstring)
			shapes.extend(subshapes)


	def generatePlan(self):
		""" convert to sequential actions.
		e.g., based on simple sequential biases.
		"""
		self.Plan = None
		assert False, "not coded"


	def shapes2strokes(self, shapes):

		def evaluateShape(shape, params):
			""" 
			- shape, is string name
			- params, different options:
			--- list, will be passed into transform in order.
			--- dict, will be passed into transform as kwargs.
			--- params order: [x=0, y=0, sx=1, sy=1, th=0, order="trs"]
			--- Note: can pass in None to get defaults.
			=== NOTES
			- now line is centered at 0,0. Original primitives centered at (0.5,0)
			=== RETURNS
			[np.array]
			"""
			def transform(p, x=None, y=None, sx=None, sy=None, theta=None, order=None):
				""" outputs stroke """
				T = Prim.makeAffine2(x, y, sx, sy, theta, order)
				return Prim._tform(p, T)

			if shape=="line":
				p = Prim.transform(Prim._line, x=-0.5)
			elif shape=="circle":
				p = Prim._circle
			else:
				print(shape)
				assert False, "not coded"

			if isinstance(params, list):
				return transform(p, *params)
			elif isinstance(params, dict):
				return transform(p, **params)
			else:
				print(params)
				assert False

		strokes =[]
		for sh in shapes:
			s = sh[0]
			if len(sh)==1:
				params = []
			else:
				params = sh[1]
			strokes.extend(evaluateShape(s, params))

		return strokes

