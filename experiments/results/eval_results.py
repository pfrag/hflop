import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def read_values_from_file(file_path):
    with open(file_path, 'r') as file:
        values = file.readlines()
    # Strip newline characters and convert to float
    values = [float(value.strip()) for value in values]
    return values

def calculate_average(values):
    if not values:
        return 0
    return sum(values) / len(values)



directories = ['./non_hier', './hier',  './hflop']
dirs = ['Non-Hierarchical Benchmark', 'Hierarchical Benchmark', 'HFLOP']
file_names = ['latencies.txt', 'calculations.txt']

data = []

# Read values from each file in each directory and sum them
for i in range(len(directories)):
    values1 = read_values_from_file(f'{directories[i]}/latencies.txt')
    values2 = read_values_from_file(f'{directories[i]}/calculations.txt')
    # Sum the values pairwise
    total_sum = [x + y for x, y in zip(values1, values2)]
    aver=calculate_average(total_sum)
    print(aver)
    # Append the total sum to the data list with the corresponding directory
    data.append({'directory': dirs[i], 'value': total_sum})
# Convert the data list to a pandas DataFrame
df = pd.DataFrame(data)
# Explode the 'value' column to create separate rows for each value
df = df.explode('value')
# Reset index after exploding
df.reset_index(drop=True, inplace=True)
# Set the size of the figure
plt.figure(figsize=(12, 12))  # Width: 12, Height: 12

# Create a boxplot
sns.boxplot(data=df, x='directory', y='value')

# Rotate x-axis labels
plt.xticks(rotation=10, fontsize=20)  # Rotate x-axis labels and set fontsize

# Set the fontsize of the y-axis tick labels
plt.yticks(fontsize=20)

# Add title and labels with fontsize
# plt.title('', fontsize=18)
plt.xlabel('', fontsize=1)
plt.ylabel('Processing times (s)', fontsize=20)

# Display the boxplot
plt.savefig("Latency")
plt.show()