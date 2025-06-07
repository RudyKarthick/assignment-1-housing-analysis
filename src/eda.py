import pandas as pd
import matplotlib.pyplot as plt


# Load cleaned data
df = pd.read_csv("housing_cleaned_final.csv")
print("Loaded cleaned dataset for EDA.")


print("Summary stats for features:")
print(df.describe())
df.describe().to_csv("eda_summary_stats.csv")



plt.hist(df['median_house_value'], bins=50, edgecolor='black')
plt.title("Distribution of Median House Value")
plt.ylabel("Number of Houses")
plt.xlabel("House Value ($)")

plt.grid(True)
plt.tight_layout()
plt.savefig("eda_median_house_value_hist.png")
print("Saved histogram to 'eda_median_house_value_hist.png'")
plt.show()