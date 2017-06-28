from pandas_datareader import data
from datetime import datetime
import matplotlib.pyplot as plt

# df = data.DataReader("KRX:KOSPI", "google")
#
# print(df)
#
# ax = df.Close.plot()
# ax.set_title("KOSPI 2016")
# ax.set_ylabel("Index")
# ax.set_xlim("2016-01-01", "2016-11-15")



start = datetime(2017,5,1)
end = datetime.now()

KA = data.DataReader('KRX:003490','google', start, end)

print(KA)

KA[['Close','Open']].plot()
plt.show()
