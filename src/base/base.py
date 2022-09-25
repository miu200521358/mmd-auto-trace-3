from enum import Enum
from typing import TypeVar

import _pickle as cPickle  # type: ignore

from base.logger import parse2str


class Encoding(Enum):
    UTF_8 = "utf-8"
    UTF_16_LE = "utf-16-le"
    SHIFT_JIS = "shift-jis"
    CP932 = "cp932"


class BaseModel:
    """基底クラス"""

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return parse2str(self)

    def copy(self):
        return cPickle.loads(cPickle.dumps(self, -1))


TBaseModel = TypeVar("TBaseModel", bound=BaseModel)
