import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


# Load cleaned data
df = pd.read_csv("housing_cleaned_final.csv")
print("Loaded cleaned dataset for EDA.")

print("Summary stats for features:")
print(df.describe())
df.describe().to_csv("eda_summary_stats.csv")


# -----Histogram-----
plt.hist(df['median_house_value'], bins=50, edgecolor='black')
plt.title("Distribution of Median House Value")
plt.ylabel("Number of Houses")
plt.xlabel("House Value ($)")

plt.grid(True)
plt.tight_layout()
plt.savefig("eda_median_house_value_hist.png")
print("Saved histogram to 'eda_median_house_value_hist.png'")
plt.show()




# -----Heatmap-----
# Source: https://commons.wikimedia.org/wiki/File:Map_of_California_outline.svg#filelinks
img = mpimg.imread("Map_of_California_outline.png")
print("Loaded California map.")

fig, ax = plt.subplots(figsize=(10,8))
# Lon & Lat bounds: https://en.wikipedia.org/wiki/California 
# Latitude	32°32′ N to 42° N = 32.533...° to 42°
# Longitude	114°8′ W to 124°26′ W = -114.133...° to -124.433...°
ax.imshow(img, extent=[-124.433, -114.133, 32.533, 42], aspect='auto')

sns.kdeplot(
    x=df['longitude'],
    y=df['latitude'],
    cmap='inferno',
    fill=True,
    alpha=0.5,
    ax=ax
)

plt.title("California Housing Data Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True)
plt.tight_layout()
plt.savefig("eda_california_heatmap.png")
print("Saved geographic heatmap to 'eda_california_heatmap.png'")
plt.show()

# -----Scatterplot-----
sns.scatterplot(data=df, x='median_income', y='median_house_value')
plt.title("House Value and Median Income")
plt.xlabel("Income ($10k)")
plt.ylabel("House Value ($)")
plt.tight_layout()
plt.savefig("eda_house_value_and_median_income_scat.png")
plt.show()

# -----Bar Plot-----
mean = df.groupby('housing_median_age')['median_house_value'].mean()
mean.plot(kind='bar')
plt.title("Mean House Value & Housing Median Age")
plt.xlabel("Median House Age")
plt.ylabel("Median House Value ($)")
plt.tight_layout()
plt.savefig("eda_house_value_and_housing_age_bar.png")
plt.show()