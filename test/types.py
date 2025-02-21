from enum import Enum


class ModelType(Enum):
	ControlNet = 1
	SmartControl = 2
	ControlAttend = 3

	@classmethod
	def str2enum(cls, model: str):
		assert model in cls._member_names_, f"The model type should be one of {cls._member_names_}, but you gave {model}"

		if model == "ControlNet":
			modelType = ModelType.ControlNet
		elif model == "SmartControl":
			modelType = ModelType.SmartControl
		elif model == "ControlAttend":
			modelType = ModelType.ControlAttend
		return modelType

class ConflictDegree(Enum):
	no_conflict = 0
	mild_conflict = 1
	significant_conflict = 2

	@classmethod
	def str2enum(cls, model: str):
		assert model in cls._member_names_, f"The model type should be one of {cls._member_names_}, but you gave {model}"
		return ConflictDegree[model]
