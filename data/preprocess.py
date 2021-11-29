import json
import pandas as pd
df = pd.read_csv("train_subset_emotion.csv", error_bad_lines=False, warn_bad_lines=False) 

df["prompt_with_emotion"] = df["prompt"] + ", " + df["context"]
df["response_with_emotion"] = df["utterance"] + ", " + df["response_context"]
dff = df[["prompt_with_emotion", "response_with_emotion"]]
grouped_df = dff.groupby("prompt_with_emotion")

grouped_lists = grouped_df["response_with_emotion"].apply(list)
grouped_lists = grouped_lists.reset_index()


compression_opts = dict(method=None,
                         archive_name='grouped_train.csv')  
grouped_lists.to_csv('grouped_train.csv', index=False,
          compression=compression_opts) 

df3 = grouped_lists
result = df3.to_json(orient="split")
parsed = json.loads(result)


lst = parsed['data']
output = []
for i in lst:
    prompt = i[0].split(", ")
    responses = []
    for j in i[1]:
        resp = j.split(", ")
        responses.append(resp)
    lst = [prompt, responses]
    output.append(lst)


with open("data", "w") as f:
    f.write(str(output))




