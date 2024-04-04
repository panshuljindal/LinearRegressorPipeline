import sys

def errorMessageDetails(error, errorDetail:sys):
    _, _, exc_tb = errorDetail.exc_info()
    errorMessage = "Erro occurred in python script name [{0}] at line number [{1}] with error message [{2}]".format(exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno, error)
    return errorMessage

class CustomException(Exception):
    def __init__(self, errorMessage, errorDetail:sys):
        super().__init__(errorMessage)
        self.errorMessage = errorMessageDetails(errorMessage, errorDetail)
    
    def __str__(self):
        return self.errorMessage
        
