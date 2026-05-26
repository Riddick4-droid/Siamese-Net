class ProjectException(Exception):
    """
    Base exception for all custom errors in this project.
    Inherits from the built-in Exception so standard error handling works.
    """
    pass

#you can add specific types of exceptions for the project
#like the ones for data downloading, ingestion, preprocessing, training ,etc

class DataIngestionException(ProjectException):
    pass
class ModelTrainingException(ProjectException):
    pass
class ModelEvaluationException(ProjectException):
    pass
