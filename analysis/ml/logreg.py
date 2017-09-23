from sklearn import linear_model, metrics
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
def logreg(x, y, output_path, test_data=None):

  logreg = linear_model.LogisticRegression(class_weight='balanced')
  accuracy = None
  for i in range(2):
    logreg.fit(x, y)
  with open(output_path, "w") as f:
    pickle.dump(logreg, f)
  if test_data:
     pred = logreg.predict(test_data[0])
     accuracy = np.mean(np.square(test_data[1] - pred))
     print("Loss:", accuracy)
     print("R^2:", metrics.r2_score(test_data[1], pred))
     print("Test Accuracy:", logreg.score(test_data[0], test_data[1]))
     print("Training Accuracy:", logreg.score(x, y))
     scores = cross_val_score(logreg, x, y)
     print("Cross VAL SCORE:", scores.mean())
  print("Wrote trained model")
  return accuracy
