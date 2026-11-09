"""Pre-fit engineering correction: bind legacy verifier name to identical package SHA checker."""
import sys, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import train
train.verify_freeze = train.verify_package
if __name__ == '__main__':
    try:
        train.main()
    except BaseException as ex:
        train.write(train.ROOT/'artifacts/FIT_FAILURE.json',dict(error=repr(ex),traceback=traceback.format_exc()))
        raise
