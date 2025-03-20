import pandas as pd
import json
with open('tmp.json', 'r') as file:
    data_dict = json.load(file)


## flatten nested json and extract specific fields
df = pd.json_normalize(data_dict['employees'])    
# df = df.explode("projects")

## count the number of projects per employee
df['num_projects']=df['projects'].apply(lambda x: len(x))
print(df['num_projects'])
print(df)
## filter employees with ongoing projects
def numActiveProjects(projects)->int:
    num =0
    for proj in projects:
        if proj['status']=='ongoing':
            num+=1
    return num

df['num_active_projects']=df['projects'].apply(numActiveProjects)
print(df['num_active_projects'])