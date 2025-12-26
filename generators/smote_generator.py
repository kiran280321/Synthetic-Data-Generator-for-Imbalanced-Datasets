from imblearn.over_sampling import SMOTE
class SMOTEGenerator:
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)
    def generate(self, X, y):
        return self.smote.fit_resample(X, y)
