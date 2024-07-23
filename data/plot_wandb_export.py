import pandas as pd
import matplotlib.pyplot as plt
# Step 1: Read the CSV file
csv_file_path = 'non_hier_10_ep.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Step 2: Process the Data
steps = data['Step']
clients = [col for col in data.columns if 'global loss' in col and '__' not in col]
# Step 3: Create the Plot
plt.figure(figsize=(10, 7))
cmap = plt.get_cmap('tab20')
for i,client in enumerate(clients):
    mean_loss = data[client]
    min_loss = data[f'{client}__MIN']
    max_loss = data[f'{client}__MAX']

    color = cmap(i)
    print(client[7:-14])
    plt.plot(steps/2, mean_loss, label="("+client[7]+")"+client[8:-14], color=color)
    plt.fill_between(steps/2, min_loss, max_loss, color=color, alpha=0.2)

# Adding labels and title
plt.xlabel('Aggregation round', fontsize=20)
plt.ylabel('MSE loss', fontsize=20)
plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

# plt.title('Global loss for each client', fontsize=24)
plt.legend(ncol=2, bbox_to_anchor=(1., 1), fontsize=12)
plt.grid(True)

# Show the plot
plt.savefig("non_hier_10ep.png")
plt.show()
