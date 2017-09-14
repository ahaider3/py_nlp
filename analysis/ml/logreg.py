from sklearn import linear_model, metrics
import pickle
import numpy as np

def logreg(x, y, output_path, test_data=None):

  logreg = linear_model.LogisticRegression(class_weight='balanced')
  accuracy = None

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

  print("Wrote trained model")
  return accuracy
