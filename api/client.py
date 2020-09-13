import requests
import json
import pandas as pd

# test sample
df = pd.read_csv('./data/test.csv')
test_sample = df.sample(2).reset_index(drop=True)


url = 'http://127.0.0.1:5000/pred'
data = test_sample.to_json(orient='records')

r = requests.post(url,json=data)

# print(r.text)
print(r.json())


# experiment
# data={'foo1':'bar1','foo2':['bar2.1','bar2.2']}
# r = requests.post(url,json=json.dumps(data))