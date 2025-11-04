import numpy as np
import shap
from sklearn.tree import DecisionTreeClassifier
from treeshap import score

if __name__ == "__main__":
    X, y = shap.datasets.adult(n_points=1000)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    weighted_node_samples = clf.tree_.weighted_n_node_samples
    values = clf.tree_.value

    result = score(np.array(X), children_left, children_right, feature, threshold, values)
    result_predict_proba = clf.predict_proba(X)
    np.testing.assert_allclose(np.squeeze(result), result_predict_proba, rtol=1e-9)
    print("In Python: ", result)
