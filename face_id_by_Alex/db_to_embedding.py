import cv2
from deepface import DeepFace
import pandas as pd

db = ("anton.jpg", "goat.jpg", "obama.jpg", "patron.jpg", "roshen.jpg", "the_kingo.jpg")
# images = [cv2.imread(f'../face_id_by_Anton/data/{known_face}') for known_face in db]

embeddings = dict()
for inx_, known_face in enumerate(db):
    embedding = DeepFace.to_embedding(f"../face_id_by_Anton/data/{known_face}", model_name="Facenet")
    embeddings[known_face] = embedding[0]["embedding"]

print(db)
for known_face_x in embeddings.keys():
    print(known_face_x, end="\t")
    for known_face_y in embeddings.keys():
        print(DeepFace.veriby_by_embeddins(
            embeddings[known_face_x],
            embeddings[known_face_y],
            model_name="Facenet",
        ), end="\t")
    print()


df = pd.DataFrame(embeddings)
df.to_csv("./data_embeddings/data_embeddings_by_Facenet.csv", index=False)