import pandas as pd
df = pd.read_csv("datasets\\train-item-views.csv", header=0)
df.columns = ['session_id;user_id;item_id;timeframe;eventdate']
print(df.head(5))
df.to_csv("datasets\\train-item-views.dat", index=False)