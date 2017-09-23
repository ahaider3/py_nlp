from sklearn import metrics
import pickle
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import cross_val_score

def random_forest(x, y, output_path, test_data=None):
  for k in [1000]:
    model = ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=-1, n_estimators=k)
    test_accuracy = None
    print("Starting Training")
    model.fit(x, y)
    with open(output_path+str(k), "w") as f:
      pickle.dump(model, f)
    if test_data:
      pred = model.predict(test_data[0])
      accuracy = np.mean(np.square(test_data[1] - pred))
      print("Loss:", accuracy)
      print("R^2:", metrics.r2_score(test_data[1], pred))
      test_accuracy = model.score(test_data[0], test_data[1])
      print("Test Accuracy:", test_accuracy)
      print("Training Accuracy:", model.score(x, y))
      scores = cross_val_score(model, x, y)
      print("CROSS VAL SCORE:", scores.mean())
 
    print("Wrote trained model")
  return test_accuracy

def adaboost(x, y, output_path, test_data=None):
  for k in [500, 1000, 100]:
    model = ensemble.AdaBoostClassifier(n_estimators=k)
    accuracy = None
    print("Starting Training")
    model.fit(x, y)
    with open(output_path+str(k), "w") as f:
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
