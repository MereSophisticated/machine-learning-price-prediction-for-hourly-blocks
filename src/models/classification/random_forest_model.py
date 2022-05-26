import matplotlib.pyplot as plt
import shap
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from src.data_analysis.data_retrieval import get_prepared_data

X, y, X_train, X_test, y_train, y_test = get_prepared_data(simple=True, labeled=True)


model = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, predictions))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.tight_layout()
plt.show()


