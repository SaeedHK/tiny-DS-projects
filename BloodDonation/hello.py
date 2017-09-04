import pandas as pd

df = pd.read_csv('Train.csv' , index_col = 0)

col = list(df.columns)

# We have
# col = ['Months since Last Donation', 'Number of Donations', 'Total Volume Donated (c.c.)', 'Months since First Donation', 'Made Donation in March 2007']

df['Average time between two giving blood'] = (df[col[3]] - df[col[0]])/ df[col[1]]

col.append('Average time between two giving blood')

df['Last time bigger than average'] = df[col[0]] > df[col[5]]

print(df.info())
