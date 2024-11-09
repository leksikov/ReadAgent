from pydantic import BaseModel
import numpy as np


class Episode(BaseModel):
    content: str
    id: int
    gist: None | str = None
    vector: np.ndarray = None

    class Config:
        arbitrary_types_allowed = True
