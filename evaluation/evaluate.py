from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
def evaluate_model(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    return {
        "report": classification_report(y_test, preds, output_dict=True),
        "roc_auc": roc_auc_score(y_test, probs)
    }
