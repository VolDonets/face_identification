import pandas as pd

df = pd.read_csv("./data_embeddings/data_embeddings_by_Facenet.csv")

columns = ["user"]
for i in range(128):
    columns.append(f"p{i}")

df_new = pd.DataFrame(columns=columns)

for col_ in df.columns:
    record_df = {"user": [col_, ]}
    for i, var_ in enumerate(df[col_].tolist()):
        record_df[f"p{i}"] = [var_, ]

    df_new = pd.concat([df_new, pd.DataFrame(record_df)])

df_new.to_csv("./data_embeddings/data_embeddings_by_Facenet_rot.csv", index=False)
