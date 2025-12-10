import os

class ResultsStore:
    def __init__(self, root_path) -> None:
        os.makedirs(root_path, exist_ok=True)
        self.root_path = root_path
    
    def store(self, result, name):
        pass
        