from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)
def devide_numer(a,b):
    try:
        result = a/b
        logger.info ("divide 2 numbers!")
        return result
    except Exception as e:
        logger.error ("Error occured!")
        raise CustomException("Custom Error zero!", sys)

if __name__=="__main__":
    try:
        logger.info("Starting main program")
        devide_numer(10,0)
    except CustomException as ce:
        logger.error(str(ce))