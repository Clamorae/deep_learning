from sklearn.datasets import fetch_olivetti_faces


data = fetch_olivetti_faces()
faces = data.images
print(faces)