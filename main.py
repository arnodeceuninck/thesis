import pandas as pd

scores = pd.DataFrame()

for i, seed in enumerate(range(5)):
    seed_nr = 8
    value = 4
    combo = 8+4

    index  =len(scores)
    scores.loc[index, 'seed'] = seed

    if i % 2 == 0:
        scores.loc[index, 'seed_nr'] = seed_nr

    scores.loc[index, 'value'] = value
    scores.loc[index, 'combo'] = combo

print(scores)

