
from .reader import InputReader

class SurrogateModel:
    def __init__(self, name="SurrogateModel"):
        self.name = name

    def fit(self, reader: InputReader):
        raise NotImplementedError

    def predict(self, reader: InputReader):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
