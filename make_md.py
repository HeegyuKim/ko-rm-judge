import pandas as pd
from glob import glob
from tqdm import tqdm
import jsonlines
import os



def main():
    files = glob("data/**/*.json")
    total = []

    for file in tqdm(list(files)):
        with jsonlines.open(file) as fin:
            for line in fin:
                if 'score' not in line:
                    continue

                dirname = os.path.dirname(file).replace("data/", "")
                filename = os.path.basename(file).replace(".json", "")

                total.append({
                    'model': dirname,
                    'testset': filename,
                    'score': line['score']
                })

    df = pd.DataFrame(total)
    df = df.groupby(['model', 'testset']).agg({"score": ["mean", "std", "count"]}).reset_index()
    df.columns = ["-".join(c) if c[1] else c[0] for c in df.columns]

    print(df)

    with open("all_results.md", "w") as fout:
        for testset in df.testset.unique():
            subset = df[df.testset == testset].drop("testset", axis=1).sort_values("score-mean", ascending=False)

            fout.write(f"## {testset}\n\n")
            fout.write(subset.to_markdown(index=False))
            fout.write("\n\n")





if __name__ == '__main__':
    main()