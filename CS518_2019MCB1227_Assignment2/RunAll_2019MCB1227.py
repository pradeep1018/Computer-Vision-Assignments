import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

def calculate_gradient(image):
    gx = np.zeros(shape=(image.shape[0],image.shape[1]))
    gy = np.zeros(shape=(image.shape[0],image.shape[1]))
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2])/2
    gy[1:-1, :] = (image[2:, :] - image[:-2, :])/2
    gx[:, 0] = image[:, 1] - image[:, 0]
    gy[0, :] = image[1, :] - image[0, :]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gy[-1, :] = image[-1, :] - image[-2, :]
    return gx, gy

def calculate_hog_cell(n_orientations, magnitudes, orientations):
    bin_width = int(180/n_orientations)
    hog_cell = np.zeros(n_orientations)
    for i in range(orientations.shape[0]):
        for j in range(orientations.shape[1]):
            orientation = orientations[i, j]
            index = int(orientation/bin_width)
            hog_cell[index] += magnitudes[i, j]
    return hog_cell/(magnitudes.shape[0]*magnitudes.shape[1])

# Histogram of Oriented Gradients(HOG) for feature extraction
def hog(image, n_orientations= 9, pixels_per_cell = (8, 8), cells_per_block = (2, 2)):
    gx, gy = calculate_gradient(image)
    sy, sx = gx.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    magnitudes = np.sqrt(gx**2 + gy**2)   
    orientations = np.rad2deg(np.arctan2(gy, gx)) % 180
    n_cellsx = int(sx / cx)
    n_cellsy = int(sy / cy)
    n_blocksx = int(n_cellsx - bx) + 1
    n_blocksy = int(n_cellsy - by) + 1
    hog_cells = np.zeros((n_cellsx, n_cellsy, n_orientations))
    prev_x = 0
    for i in range(n_cellsx):
        prev_y = 0
        for j in range(n_cellsy):
            magnitudes_patch = magnitudes[prev_y:prev_y + cy, prev_x:prev_x + cx]
            orientations_patch = orientations[prev_y:prev_y + cy, prev_x:prev_x + cx]
            hog_cells[j, i] = calculate_hog_cell(n_orientations, magnitudes_patch, orientations_patch)
            prev_y += cy
        prev_x += cx
    hog_blocks = []
    for i in range(n_blocksx):
        for j in range(n_blocksy):
            hog_block = hog_cells[j:j + by, i:i + bx].ravel()
            hog_block = hog_block/np.sqrt(np.sum(hog_block**2)+1e-6) 
            hog_blocks.extend(hog_block)
    hog_blocks = np.array(hog_blocks)
    return hog_blocks

def get_features(images):
    features = []
    for image in tqdm(images):
      features.append(hog(image))
    features = np.array(features)
    return features

class KMeansClustering:
    def __init__(self, n_clusters=10, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter 
        self.n_features = None
        self.n_examples = None
        self.centroids = None
        
    def initialize_centroids(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]
    
    def create_clusters(self, X):
        clusters = [[] for _ in range(self.n_clusters)]
        for index, point in enumerate(X):
            closest_centroid = np.argmin(np.linalg.norm(point-self.centroids, axis=1)) 
            clusters[closest_centroid].append(index)
        return clusters

    def get_new_centroids(self, X, cluster):
        for index, cluster in enumerate(cluster):
            centroid = np.mean(X[cluster], axis=0)
            self.centroids[index] = centroid
        return self.centroids

    def fit(self, X):
        self.n_examples, self.n_features = X.shape
        self.initialize_centroids(X)
        for _ in range(self.max_iter):
            clusters = self.create_clusters(X)
            prev_centroids = self.centroids
            self.get_new_centroids(X, clusters)
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            diff = self.centroids - prev_centroids
            if not diff.any():
                break
        return self.centroids

def CreateVisualDictionary(features, n_clusters=10):
    kmeans = KMeansClustering(n_clusters=n_clusters)
    visualdict = kmeans.fit(features)
    return visualdict

def ComputeHistogram(visualdict, features):
    histograms = np.zeros(shape=(features.shape[0], visualdict.shape[0]))
    for index in tqdm(range(visualdict.shape[0])):
        weights = np.linalg.norm(visualdict[index]-features, axis=1)
        weights = np.exp(-weights)
        histograms[:,index] = weights
    histograms = histograms/np.expand_dims(np.sum(histograms, axis=1),axis=1) 
    return histograms

def MatchHistogram(train_histograms, test_histograms, train_labels, test_labels):
  test_labels_pred = np.zeros(test_labels.shape[0])
  for test_index in tqdm(range(test_histograms.shape[0])):
    test_labels_pred[test_index] = train_labels[np.argmin(np.linalg.norm(test_histograms[test_index]-train_histograms, axis=1))]
  return test_labels_pred

def main(n_clusters=40):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_features = get_features(train_images)
    test_features = get_features(test_images)
    visualdict = CreateVisualDictionary(train_features, n_clusters)
    train_histograms = ComputeHistogram(visualdict, train_features)
    test_histograms = ComputeHistogram(visualdict, test_features)
    test_labels_pred = MatchHistogram(train_histograms, test_histograms, train_labels, test_labels)
    classification_accuracy = accuracy_score(test_labels, test_labels_pred)
    precision = precision_score(test_labels, test_labels_pred, average=None)
    recall = recall_score(test_labels, test_labels_pred, average=None)
    print('Classification accuracy is',classification_accuracy)
    print('Precision is',precision)
    print('Recall is', recall)

main()