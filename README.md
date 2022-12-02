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
data = pd.read_excel('data.xlsx')
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
      <th>Eta</th>
      <th>Sesso</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>Unnamed: 4</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>Unnamed: 7</th>
      <th>RT</th>
      <th>TMz cicli</th>
      <th>Unnamed: 10</th>
      <th>EORMdC (percentuale)</th>
      <th>EORtot</th>
      <th>Unnamed: 13</th>
      <th>OS (mesi)</th>
      <th>PFS (mesi)</th>
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
      <th>Eta</th>
      <th>Sesso</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>RT</th>
      <th>TMz cicli</th>
      <th>EORMdC (percentuale)</th>
      <th>EORtot</th>
      <th>OS (mesi)</th>
      <th>PFS (mesi)</th>
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

corr = data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap='PiYG')
plt.show()
```


    
![png](notebook_files/notebook_7_0.png)
    


In order to more effectively apply machine learning algorithms in the future, we need to normalize a few columns.


```python
#normalize data by column
column_excluded = ['Sesso', 'MGMTmet', 'IDH1']
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
      <th>Eta</th>
      <th>Sesso</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>RT</th>
      <th>TMz cicli</th>
      <th>EORMdC (percentuale)</th>
      <th>EORtot</th>
      <th>OS (mesi)</th>
      <th>PFS (mesi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.973648</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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



We also remap the values in the 'Sesso' column to 1 and 0.


```python
data['Sesso'] = data['Sesso'].apply(lambda x: 1 if x == 1 else 0)
data
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
      <th>Eta</th>
      <th>Sesso</th>
      <th>MGMTmet</th>
      <th>IDH1</th>
      <th>VolMdCpre</th>
      <th>Vol FLAIR</th>
      <th>RT</th>
      <th>TMz cicli</th>
      <th>EORMdC (percentuale)</th>
      <th>EORtot</th>
      <th>OS (mesi)</th>
      <th>PFS (mesi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.973648</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1.270807</td>
      <td>-1.099610</td>
      <td>-1.558608</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.272214</td>
      <td>0.052293</td>
      <td>-0.171736</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>-1.223018</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.225850</td>
      <td>-1.077544</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.680710</td>
      <td>0.325920</td>
      <td>0.601076</td>
    </tr>
    <tr>
      <th>146</th>
      <td>-0.890525</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.970067</td>
      <td>-0.333332</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.111218</td>
      <td>-0.221333</td>
      <td>0.085868</td>
    </tr>
    <tr>
      <th>147</th>
      <td>-0.308664</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.604713</td>
      <td>-1.095598</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>1.008170</td>
      <td>0.873173</td>
      <td>0.343472</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0.771936</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.796518</td>
      <td>1.157098</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>-0.018474</td>
      <td>-0.403751</td>
      <td>-0.944548</td>
    </tr>
    <tr>
      <th>149</th>
      <td>-0.391787</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.551260</td>
      <td>-0.515875</td>
      <td>0.321120</td>
      <td>-0.085274</td>
      <td>0.231308</td>
      <td>0.819952</td>
      <td>0.690755</td>
      <td>-0.300538</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 12 columns</p>
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
plot_correlation(data['EORtot'], data['OS (mesi)'])
```


    
![png](notebook_files/notebook_13_0.png)
    


Let's use dimensionality reduction to see if we can find some interesting patterns in the data. This condenses the numerical information of each patient into 2 dimensions so that we can plot it.
We use the first 10 columns as input and the last 2 as color. Thus green means longer than average and red means shorter than average survival time.


```python
# UMAP on data
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(data.iloc[:, :-2])
```


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
plt.show()
```


    
![png](notebook_files/notebook_16_0.png)
    


One can easily see a color gradient: this means that the patients are in principle separable from the data available. Moreover, new patients can be located in this 2D space to predict the survival time.

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

    Epoch: 599, Train Loss: 0.2685
    Epoch: 599, Test Loss: 0.9058
    


    
![png](notebook_files/notebook_26_1.png)
    


    Finished Training
    

Let's try a clustering algorithm. (To do...)


```python
# kmeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(data.iloc[:, :-1])
```
