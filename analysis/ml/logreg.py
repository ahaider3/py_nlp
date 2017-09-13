from sklearn import linear_model
import pickle

def logreg(x, y, output_path, test_data=None):

  logreg = linear_model.LogisticRegression()
  accuracy = None
  logreg.fit(x, y)
  with open(output_path, "w") as f:
    pickle.dump(logreg, f)
  if test_data:
     accuracy = np.mean(np.square(test_data[1] - logreg.predict(test_data[0])))
     print("Accuracy:", accuracy)
  print("Wrote trained model")
  return accuracy
