from deepface.DeepFace import *


def to_embedding(
    img_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    enforce_detection=True,
    align=True,
    normalization="base",
):
    # --------------------------------
    target_size = functions.find_target_size(model_name=model_name)

    # img pairs might have many faces
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    # --------------------------------
    img_embedding_obj = represent(
        img_path=img_objs[0][0],
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend="skip",
        align=align,
        normalization=normalization,
    )

    return img_embedding_obj


def veriby_by_embeddins(
        img1_representation,
        img2_representation,
        model_name="VGG-Face",
        distance_metric="cosine",

):
    if distance_metric == "cosine":
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == "euclidean":
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == "euclidean_l2":
        distance = dst.findEuclideanDistance(
            dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    threshold = dst.findThreshold(model_name, distance_metric)
    return distance <= threshold