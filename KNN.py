import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

k = 3

class KNN:

    def __init__(self, k):
        # Load MNIST dataset from OpenML
        self.mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=True)
        
        # Convert target to integers (since they come as strings by default)
        self.data, self.target = self.mnist.data, self.mnist.target.astype(int)
        
        # Make an array of indices the size of MNIST to use for making the data sets.
        self.indx = np.random.choice(len(self.target), 70000, replace=False)
        
        # Initialize the KNN classifier
        self.classifier = KNeighborsClassifier(n_neighbors=k)

    def mk_dataset(self, size):
        """Creates a dataset of a specific size."""
        train_img = [self.data.iloc[i] for i in self.indx[:size]]
        train_img = np.array(train_img)
        train_target = [self.target.iloc[i] for i in self.indx[:size]]
        train_target = np.array(train_target)

        return train_img, train_target

    def skl_knn(self):
        """Train the KNN model and save it as 'knn.sav'."""
        # Generate datasets
        fifty_x, fifty_y = self.mk_dataset(50000)
        
        # Prepare test data (the next 10,000 images)
        test_img = [self.data.iloc[i] for i in self.indx[60000:70000]]
        test_img = np.array(test_img)
        test_target = [self.target.iloc[i] for i in self.indx[60000:70000]]
        test_target = np.array(test_target)

        # Train the model
        self.classifier.fit(fifty_x, fifty_y)

        # Make predictions on the test data
        y_pred = self.classifier.predict(test_img)
        
        # Save the model to a file
        pickle.dump(self.classifier, open('knn.sav', 'wb'))
        
        # Print the classification report
        print(classification_report(test_target, y_pred))
        print("KNN Classifier model saved as knn.sav!")

# Initialize and run the KNN model
if __name__ == "__main__":
    knn_model = KNN(k=3)
    knn_model.skl_knn()
