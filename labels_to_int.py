import csv

with open('dataset/labels.csv', 'r') as f:
  reader = csv.reader(f)
  the_list = list(reader)

#list of list to list
listone = []
index = 0
for item in the_list:
    listone.append(the_list[index][0])
    index += 1
print(listone)