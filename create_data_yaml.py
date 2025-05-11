data_yaml = """
path: dataset  
train: images/train
val: images/val

nc: 1
names: ['plate'] 
"""

with open('dataset/data.yaml', 'w') as f:
    f.write(data_yaml)
