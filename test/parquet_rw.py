import pandas as pd
from shared.io import write_parquet, read_parquet
df = pd.DataFrame({"a":[1,2,3]})
write_parquet(df, "test.parquet")
df2 = read_parquet("test.parquet")
assert df.equals(df2)