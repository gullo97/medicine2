In this notebook we start by preprocess the data and create a dataframe that we can use for further studies.  
We start by importing the necessary libraries.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

Let's take a look at the data.


```python
# read data from xlsx file
data = pd.read_excel('data_eng.xlsx')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>Unnamed: 4</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>Unnamed: 7</th>
      <th>RT</th>
      <th>TMz cycles</th>
      <th>Unnamed: 10</th>
      <th>EORMdC (%)</th>
      <th>EORtot (%)</th>
      <th>Unnamed: 13</th>
      <th>OS (months)</th>
      <th>PFS (months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>51.5</td>
      <td>172.6</td>
      <td>NaN</td>
      <td>64</td>
      <td>6</td>
      <td>NaN</td>
      <td>0.955340</td>
      <td>0.643685</td>
      <td>NaN</td>
      <td>17</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>48.2</td>
      <td>146.4</td>
      <td>NaN</td>
      <td>64</td>
      <td>6</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.890710</td>
      <td>NaN</td>
      <td>25</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>43.8</td>
      <td>11.2</td>
      <td>NaN</td>
      <td>64</td>
      <td>6</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.803571</td>
      <td>NaN</td>
      <td>23</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>82.2</td>
      <td>227.7</td>
      <td>NaN</td>
      <td>64</td>
      <td>6</td>
      <td>NaN</td>
      <td>0.959854</td>
      <td>0.631094</td>
      <td>NaN</td>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>76</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>10.2</td>
      <td>26.5</td>
      <td>NaN</td>
      <td>48</td>
      <td>6</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.845283</td>
      <td>NaN</td>
      <td>29</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



The excel file has some unnamed columns, so we will drop them.


```python
# drop Unnamed columns
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>RT</th>
      <th>TMz cycles</th>
      <th>EORMdC (%)</th>
      <th>EORtot (%)</th>
      <th>OS (months)</th>
      <th>PFS (months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>51.5</td>
      <td>172.6</td>
      <td>64</td>
      <td>6</td>
      <td>0.955340</td>
      <td>0.643685</td>
      <td>17</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>48.2</td>
      <td>146.4</td>
      <td>64</td>
      <td>6</td>
      <td>1.000000</td>
      <td>0.890710</td>
      <td>25</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>43.8</td>
      <td>11.2</td>
      <td>64</td>
      <td>6</td>
      <td>1.000000</td>
      <td>0.803571</td>
      <td>23</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>82.2</td>
      <td>227.7</td>
      <td>64</td>
      <td>6</td>
      <td>0.959854</td>
      <td>0.631094</td>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>76</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10.2</td>
      <td>26.5</td>
      <td>48</td>
      <td>6</td>
      <td>1.000000</td>
      <td>0.845283</td>
      <td>29</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



We now use seaborn to plot the corrrelation matrix.


```python
# plot correlation matrix

# corr = data.corr()
# plt.figure(figsize=(20,20))
# sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap='PiYG')
# plt.yticks(rotation=0)
# plt.xticks(rotation=45)
# plt.savefig('corr_full.png')
# plt.show()

# #plot last 2 rows of correlation matrix
corr = data.corr()
plt.figure(figsize=(24,4))
sns.heatmap(corr.iloc[-2:,:], annot=True, fmt=".2f", center=0, cmap='PiYG')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.savefig('corr_vers2.png')
plt.show()
```


    
![png](notebook_files/notebook_7_0.png)
    


In order to more effectively apply machine learning algorithms in the future, we need to normalize a few columns.


```python
#normalize data by column
column_excluded = []#['Sesso', 'MGMTmet', 'IDH1']
for column in data.columns:
    if column not in column_excluded:
        data[column] = (data[column] - data[column].mean()) / data[column].std()
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>RT</th>
      <th>TMz cycles</th>
      <th>EORMdC (%)</th>
      <th>EORtot (%)</th>
      <th>OS (months)</th>
      <th>PFS (months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.973648</td>
      <td>-0.859782</td>
      <td>-1.051313</td>
      <td>0.848122</td>
      <td>0.427323</td>
      <td>1.831101</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>-0.360291</td>
      <td>-0.686745</td>
      <td>-1.042213</td>
      <td>-0.686944</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.973648</td>
      <td>-0.859782</td>
      <td>-1.051313</td>
      <td>0.848122</td>
      <td>0.291637</td>
      <td>1.305539</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.488303</td>
      <td>-0.312542</td>
      <td>0.343472</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.641156</td>
      <td>-0.859782</td>
      <td>-1.051313</td>
      <td>0.848122</td>
      <td>0.110722</td>
      <td>-1.406522</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.073801</td>
      <td>-0.494960</td>
      <td>-0.042934</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.187552</td>
      <td>-0.859782</td>
      <td>0.944851</td>
      <td>0.848122</td>
      <td>1.689613</td>
      <td>2.936386</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>-0.300493</td>
      <td>-0.746639</td>
      <td>-0.859795</td>
      <td>-0.944548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.104429</td>
      <td>-0.859782</td>
      <td>-1.051313</td>
      <td>0.848122</td>
      <td>-1.270807</td>
      <td>-1.099610</td>
      <td>-1.558608</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.272214</td>
      <td>0.052293</td>
      <td>-0.171736</td>
    </tr>
  </tbody>
</table>
</div>



Here we can choose 2 columns and plot them against each other.


```python
def plot_correlation(column_1, column_2):
    plt.figure(figsize=(8,8))
    sns.scatterplot(x=column_1, y=column_2, data=data)
    # plot line of best fit
    sns.regplot(x=column_1, y=column_2, data=data, scatter=False, color='red')    
    plt.show()
    

# plot_correlation(data[data.columns[-4]], data['OS (mesi)'])
plot_correlation(data['EORtot (%)'], data['OS (months)'])
```


    
![png](notebook_files/notebook_11_0.png)
    


Let's use dimensionality reduction to see if we can find some interesting patterns in the data. This condenses the numerical information of each patient into 2 dimensions so that we can plot it.
We use the first 10 columns as input so that the model has no info about the last 2 columns which we want to predict.


```python
# UMAP on data
import umap
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data.iloc[:, :-2])
#plot embedding
plt.figure(figsize=(10,10))
plt.scatter(embedding[:, 0], embedding[:, 1], marker = 'x')
```




    <matplotlib.collections.PathCollection at 0x1e9d187ae08>




    
![png](notebook_files/notebook_13_1.png)
    


We can then use the last 2 columns as labels and plot the results to see if we can find any patterns.


```python
# plot embedding
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.scatter(embedding[:, 0], embedding[:, 1], c=data.iloc[:, -2], cmap='PiYG', label=data.columns[-2])
plt.legend()
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(embedding[:, 0], embedding[:, 1], c=data.iloc[:, -1], cmap='PiYG', label=data.columns[-1])
plt.colorbar()
plt.legend()
plt.savefig('umap_embedding.png')
plt.show()
```


    
![png](notebook_files/notebook_15_0.png)
    


One can easily see a color gradient: this means that the patients are in principle separable from the data available. Moreover, new patients can be located in this 2D space to predict the survival time.

But we are not done yet. We can also invert the process and identify the most important features that determine the survival time. This is done by using the built in inverse trasform of UMAP.


```python
# #save dataframe 
# data.to_csv('data_norm.csv', index=False)
```


```python
#load dataframe
data = pd.read_csv('data_norm.csv')
labels = data.iloc[:, -2]
mapper = umap.UMAP(random_state=42).fit(data.iloc[:, :-2])

# mapper.embedding_
plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=labels, cmap='PiYG')
```




    <matplotlib.collections.PathCollection at 0x1e9d1a43448>




    
![png](notebook_files/notebook_18_1.png)
    



```python
corners = np.array([
    [7, 5.5],  
    [14, 2.6], 
    [8.65, 8.65],  
    [11.5, 8.65],  
])

test_pts = np.array([
    (corners[0]*(1-x) + corners[1]*x)*(1-y) +
    (corners[2]*(1-x) + corners[3]*x)*y
    for y in np.linspace(0, 1, 5)
    for x in np.linspace(0, 1, 5)
])


```


```python
inv_transformed_points = mapper.inverse_transform(test_pts)
```


```python
from matplotlib.gridspec import GridSpec
# Set up the grid
fig = plt.figure(figsize=(12,6))
gs = GridSpec(5, 10, fig)
scatter_ax = fig.add_subplot(gs[:, :5])
digit_axes = np.zeros((5, 5), dtype=object)
for i in range(5):
    for j in range(5):
        digit_axes[i, j] = fig.add_subplot(gs[i, 5 + j])

# Use umap.plot to plot to the major axis
# umap.plot.points(mapper, labels=labels, ax=scatter_ax)
scatter_ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=labels, cmap='PiYG', s=16)
scatter_ax.set(xticks=[], yticks=[])

# Plot the locations of the text points
scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=25)

# Plot each of the generated digit images
for i in range(5):
    for j in range(5):
        digit_axes[i, j].bar(range(len(inv_transformed_points[i*5 + j])), inv_transformed_points[i*5 + j])
        digit_axes[i, j].set(xticks=[], yticks=[])
plt.savefig('umap_embedding_reversed.png')
```


    
![png](notebook_files/notebook_21_0.png)
    


Let's plot a single point transformed back to the original space.


```python
test_pts = np.array([7.1,5.6]).reshape(1, -1)
inv_transformed_points = mapper.inverse_transform(test_pts)
#plot embedding and inverse transformed points
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=labels, cmap='PiYG')
plt.scatter(test_pts[:, 0], test_pts[:, 1], c='red', marker='x', s=400)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.bar(range(len(inv_transformed_points[0])), inv_transformed_points[0])
plt.xticks(range(len(inv_transformed_points[0])), data.columns[:-2], rotation=35)
plt.savefig('umap_embedding_reversed_1.png')
plt.show()

```


    
![png](notebook_files/notebook_23_0.png)
    


To play around with this, you can use the app.py file in this repository. It is a simple streamlit app that allows to click on this 2D space and get the original features of the patient.

Plotting lines can show us how the features change with the survival time.


```python
def line_centered_at_point(point, slope, length=1, n_samples=3):
    """Return a line centered at a point with a given slope and length."""
    xs = np.linspace(point[0] - length/2, point[0] + length/2, n_samples)
    if slope == 0:
        ys = np.ones(n_samples) * point[1]
    else:
        ys = point[1] + (xs - point[0]) * slope
    return np.array([xs, ys]).T

test_pts = np.array([11,5.5])
test_pts = line_centered_at_point(test_pts, -1, length=5, n_samples=30)
inv_transformed_points = mapper.inverse_transform(test_pts)

#plot embedding and inverse transformed points
plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=labels, cmap='PiYG')
plt.scatter(test_pts[:, 0], test_pts[:, 1], c='red', marker='x', s=20)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.xticks([])
plt.yticks([])

# colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
# ax2 = plt.subplot(1,2,2, projection='3d')
# for i, point in enumerate(inv_transformed_points):
#     ax2.bar(range(len(point)), point, zs=i, zdir='y', color=colors[i%len(colors)], alpha=0.8)
# plt.xticks(range(len(inv_transformed_points[0])), data.columns[:-2], rotation=35)

ax2 = plt.subplot(1,2,2)
for i in range(len(inv_transformed_points[0])):
    ax2.plot(inv_transformed_points[:, i], label=data.columns[i])
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xticks([])
plt.xlabel('<----- Top-Left to Bottom-Right ----->')



```




    Text(0.5, 0, '<----- Top-Left to Bottom-Right ----->')




    
![png](notebook_files/notebook_26_1.png)
    


Let's now create a neural network model to try and fit the data.


```python
import torch
import torch.nn as nn

L_SIZE = 5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.fc1 = nn.Linear(10, L_SIZE)
        self.fc2 = nn.Linear(L_SIZE, L_SIZE)
        self.fc3 = nn.Linear(L_SIZE, L_SIZE)
        self.fc4 = nn.Linear(L_SIZE, L_SIZE)
        self.fc5 = nn.Linear(L_SIZE, L_SIZE)
        self.fc6 = nn.Linear(L_SIZE, L_SIZE)
        self.fc7 = nn.Linear(L_SIZE, 1)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.to(self.device)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        # x = self.fc5(x)
        # x = self.relu(x)
        # x = self.fc6(x)
        # x = self.relu(x)
        x = self.fc7(x)
        # x = self.sigmoid(x)
        
        return x
    def train_model(self, train_loader, test_loader, epochs=100, lr=0.001, PLOT_INTERVAL=150):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs).view(-1,1)
                loss = criterion(outputs, labels)
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self(inputs).view(-1,1)
                    test_loss = criterion(outputs, labels)
                    test_losses.append(test_loss.item())
            if((epoch % PLOT_INTERVAL == 0 and epoch!=0) or epoch == epochs-1):
                print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, loss.item()))
                print('Epoch: {}, Test Loss: {:.4f}'.format(epoch, test_loss.item()))
                plt.figure(figsize=(8,8))
                plt.plot(train_losses)
                plt.plot(test_losses)
                plt.legend(['train loss', 'test loss'])
                plt.show()
        print('Finished Training')
```

The model need a dataset class to work with.


```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.Tensor([self.data.iloc[idx, :-2]]), torch.Tensor([self.data.iloc[idx, -1]])
    def train_test_split(self, test_size=0.2):
        train_data = self.data.sample(frac=1-test_size, random_state=0)
        test_data = self.data.drop(train_data.index)
        return train_data, test_data
```

Now we split data into training and test sets.


```python
# torch.Tensor([data.iloc[1, :-1].values]).to('cuda:0')
dataset = CustomDataset(data)
train_data, test_data = dataset.train_test_split()
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
```

And we define the Dataloaders.


```python
# import dataloader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=150, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)

```

We are now ready to train the model.


```python
model = Net()
model.train_model(train_loader, test_loader, epochs=600, lr=0.001, PLOT_INTERVAL=2000)

#save model state_dict
# torch.save(model.state_dict(), 'model.pth')
```

    Epoch: 599, Train Loss: 0.3257
    Epoch: 599, Test Loss: 0.8952
    


    
![png](notebook_files/notebook_36_1.png)
    


    Finished Training
    

Let's try a clustering algorithm. (To do...)


```python
# kmeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(data.iloc[:, :-1]) # careful with the last column
```
