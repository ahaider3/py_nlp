from sklearn import svm as SVM
from sklearn import metrics
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score

def svm(x, y, output_path, test_data=None):

  model = SVM.SVC(class_weight='balanced')
  accuracy = None
  for i in range(2):
    model.fit(x, y)
  with open(output_path, "w") as f:
    pickle.dump(model, f)
  if test_data:
     pred = model.predict(test_data[0])
     accuracy = np.mean(np.square(test_data[1] - pred))
     print("Loss:", accuracy)
     print("R^2:", metrics.r2_score(test_data[1], pred))
     print("Test Accuracy:", model.score(test_data[0], test_data[1]))
     print("Training Accuracy:", model.score(x, y))
     scores = cross_val_score(model, x, y)
     print("CROSS VAL SCORE:", scores.mean())
  print("Wrote trained model")
  return accuracy
