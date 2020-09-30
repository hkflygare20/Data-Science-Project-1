

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import scipy.stats as stats
%matplotlib inline
df = pd.read_csv("POLITICAL_FAVORABLES_RCP_icod_20200317.csv")
df.head()
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
      <th>SOURCE</th>
      <th>SURVERYPERIOD</th>
      <th>DATEFORMATTED</th>
      <th>SAMPLE</th>
      <th>FAVORABLE</th>
      <th>UNFAVORABLE</th>
      <th>MARGIN</th>
      <th>CANDIDATE</th>
      <th>DAYS_TO_GENERAL_ELECTION</th>
      <th>ELECTION_YEAR</th>
      <th>ELECTION_DATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ABC/Wash Post</td>
      <td>4/18 - 4/22</td>
      <td>4/22/2012</td>
      <td>1008 A</td>
      <td>65</td>
      <td>27</td>
      <td>38</td>
      <td>HRC</td>
      <td>1661</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABC/Wash Post</td>
      <td>11/28 - 12/2</td>
      <td>12/2/2012</td>
      <td>RV</td>
      <td>65</td>
      <td>31</td>
      <td>34</td>
      <td>HRC</td>
      <td>1437</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABC/Wash Post</td>
      <td>1/16 - 1/20</td>
      <td>1/20/2013</td>
      <td>RV</td>
      <td>65</td>
      <td>30</td>
      <td>35</td>
      <td>HRC</td>
      <td>1388</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABC/Wash Post</td>
      <td>6/19 - 6/23</td>
      <td>6/23/2013</td>
      <td>RV</td>
      <td>60</td>
      <td>36</td>
      <td>24</td>
      <td>HRC</td>
      <td>1234</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABC/Wash Post</td>
      <td>1/20 - 1/23</td>
      <td>1/23/2014</td>
      <td>873 RV</td>
      <td>56</td>
      <td>41</td>
      <td>15</td>
      <td>HRC</td>
      <td>1020</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(['SAMPLE','DATEFORMATTED','SURVERYPERIOD','ELECTION_DATE','UNFAVORABLE','MARGIN'],axis=1)
Hilary = df.loc[df['CANDIDATE'] == 'HRC']
Trump = df.loc[df['CANDIDATE'] == 'TRUMP']
Biden = df.loc[df['CANDIDATE'] == 'BIDEN']
Bernie = df.loc[df['CANDIDATE'] == 'BERNIE']
```


```python
Trump20 = Trump.loc[Trump['ELECTION_YEAR']==2020]
Trump16 = Trump.loc[Trump['ELECTION_YEAR']==2016]
Bernie20 = Bernie.loc[Bernie['ELECTION_YEAR']==2020]
Bernie16= Bernie.loc[Bernie['ELECTION_YEAR']==2016]
```


```python
Hilary = Hilary.drop(['CANDIDATE','ELECTION_YEAR',], axis=1)
Trump20 = Trump20.drop(['CANDIDATE','ELECTION_YEAR',], axis=1)
Trump16 = Trump16.drop(['CANDIDATE','ELECTION_YEAR',], axis=1)
Bernie20 = Bernie20.drop(['CANDIDATE','ELECTION_YEAR',], axis=1)
Bernie16 = Bernie16.drop(['CANDIDATE','ELECTION_YEAR',], axis=1)
Biden = Biden.drop(['CANDIDATE','ELECTION_YEAR',], axis=1)
print("done")
```

    done



```python
#Hilary polls
HFarLeft = Hilary.loc[(Hilary['SOURCE']=='CNN/Opinion Research')|(Hilary['SOURCE']=='CNN/ORC')]
HLeft= Hilary.loc[(Hilary['SOURCE']=='CNN')|(Hilary['SOURCE']=='NBC/WSJ')|(Hilary['SOURCE']=='CBS News/NY Times')|(Hilary['SOURCE']=='CBS News')|(Hilary['SOURCE']=='ABC/Wash Post')|(Hilary['SOURCE']=='Economist/YouGov')|(Hilary['SOURCE']=='The Atlantic/PRRI')]
HCenter = Hilary.loc[(Hilary['SOURCE']=='Associated Press/Gfk')|(Hilary['SOURCE']=='Bloomberg')|(Hilary['SOURCE']=='USA Today/Gallup')|(Hilary['SOURCE']=='USA Today/Suffolk')]
HRight = Hilary.loc[Hilary['SOURCE']=='FOX News']
#HFarRight = Hilary.loc[Hilary['SOURCE']==']
HIPolls = Hilary.loc[(Hilary['SOURCE']=='Gallup')|(Hilary['SOURCE']=='Gravis')|(Hilary['SOURCE']=='IDB/TIPP')|(Hilary['SOURCE']=='PPP (D)')|(Hilary['SOURCE']=='Pew Research')|(Hilary['SOURCE']=='Rasmussen Resports')|(Hilary['SOURCE']=='Reuters/Ipsos')]
HUPolls = Hilary.loc[(Hilary['SOURCE']=='McClatchy/Marist')|(Hilary['SOURCE']=='GWU/Battlegroud')|(Hilary['SOURCE']=='Monmouth')|(Hilary['SOURCE']=='Politico/GWU/Battleground')|(Hilary['SOURCE']=='Quinnipiac')]

def Clean_data(df1):
    
    df1 = df1.drop(['SOURCE','DAYS_TO_GENERAL_ELECTION'],axis=1)
    df1 = df1.reset_index(drop=True)
    
    return df1

HFarLeft = Clean_data(HFarLeft)
HFarLeft= HFarLeft.rename(columns={'FAVORABLE':'FarLeft'})
HLeft= Clean_data(HLeft)
HLeft= HLeft.rename(columns={'FAVORABLE':'Left'})
HCenter= Clean_data(HCenter)
HCenter= HCenter.rename(columns={'FAVORABLE':'Center'})
HRight = Clean_data(HRight)
HRight= HRight.rename(columns={'FAVORABLE':'Right'})
HIPolls= Clean_data(HIPolls)
HIPolls= HIPolls.rename(columns={'FAVORABLE':'Polls'})
HUPolls= Clean_data(HUPolls)
HUPolls= HUPolls.rename(columns={'FAVORABLE':'Uni Polls'})

'''
plt.plot(HFarLeft['DAYS_TO_GENERAL_ELECTION'],HFarLeft['FAVORABLE'], 'bd')
plt.plot(HLeft['DAYS_TO_GENERAL_ELECTION'],HLeft['FAVORABLE'], 'cd')
plt.plot(HCenter['DAYS_TO_GENERAL_ELECTION'],HCenter['FAVORABLE'],'md')
plt.plot(HRight['DAYS_TO_GENERAL_ELECTION'],HRight['FAVORABLE'],'rd')
plt.plot(HIPolls['DAYS_TO_GENERAL_ELECTION'],HIPolls['FAVORABLE'],'C1d')
plt.plot(HUPolls['DAYS_TO_GENERAL_ELECTION'],HUPolls['FAVORABLE'],'kd')
#plt.plot(NBC_H['DAYS_TO_GENERAL_ELECTION'],NBC_H['FAVORABLE'],'ro')
plt.legend()
'''

```




    "\nplt.plot(HFarLeft['DAYS_TO_GENERAL_ELECTION'],HFarLeft['FAVORABLE'], 'bd')\nplt.plot(HLeft['DAYS_TO_GENERAL_ELECTION'],HLeft['FAVORABLE'], 'cd')\nplt.plot(HCenter['DAYS_TO_GENERAL_ELECTION'],HCenter['FAVORABLE'],'md')\nplt.plot(HRight['DAYS_TO_GENERAL_ELECTION'],HRight['FAVORABLE'],'rd')\nplt.plot(HIPolls['DAYS_TO_GENERAL_ELECTION'],HIPolls['FAVORABLE'],'C1d')\nplt.plot(HUPolls['DAYS_TO_GENERAL_ELECTION'],HUPolls['FAVORABLE'],'kd')\n#plt.plot(NBC_H['DAYS_TO_GENERAL_ELECTION'],NBC_H['FAVORABLE'],'ro')\nplt.legend()\n"




```python
Hdata = [HFarLeft['FarLeft'],HLeft['Left'],HCenter['Center'],HRight['Right'],HIPolls['Polls'],HUPolls['Uni Polls']]

Hheaders = ['Left','Lean Left','Center','Lean Right','Ind Polls','Uni Polls',]

dfH = pd.concat(Hdata, axis=1, keys=Hheaders)

dfH.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3d054470>




![png](output_5_1.png)



```python
# stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(HFarLeft['FarLeft'],HLeft['Left'],HCenter['Center'],HRight['Right'],HIPolls['Polls'],HUPolls['Uni Polls'])
print(fvalue, pvalue)

```

    2.851553823113297 0.015371885191054327



```python
#Trump '16 polls
T16FarLeft = Trump16.loc[(Trump16['SOURCE']=='CNN/Opinion Research')|(Trump16['SOURCE']=='CNN/ORC')]
T16Left= Trump16.loc[(Trump16['SOURCE']=='NBC/WSJ')|(Trump16['SOURCE']=='ABC/Wash Post')|(Trump16['SOURCE']=='CBS News')|(Trump16['SOURCE']=='CBS News/NY Times')|(Trump16['SOURCE']=='Economist/YouGov')|(Trump16['SOURCE']=='The Atlantic/PRRI')]
T16Center = Trump16.loc[(Trump16['SOURCE']=='Associated Press/Gfk')|(Trump16['SOURCE']=='Bloomberg')|(Trump16['SOURCE']=='USA Today/Suffolk')]
T16Right = Trump16.loc[Trump16['SOURCE']=='FOX News']
#T16FarRight = Trump16.loc[Trump16['SOURCE']==']
T16IPolls = Trump16.loc[(Trump16['SOURCE']=='Gallup')|(Trump16['SOURCE']=='Gravis')|(Trump16['SOURCE']=='IDB/TIPP')|(Trump16['SOURCE']=='PPP (D)')|(Trump16['SOURCE']=='Rasmussen Resports')|(Trump16['SOURCE']=='Reuters/Ipsos')]
T16UPolls = Trump16.loc[(Trump16['SOURCE']=='McClatchy/Marist')|(Trump16['SOURCE']=='GWU/Battlegroud')|(Trump16['SOURCE']=='Monmouth')|(Trump16['SOURCE']=='Quinnipiac')]

T16FarLeft = Clean_data(T16FarLeft)
T16FarLeft= T16FarLeft.rename(columns={'FAVORABLE':'FarLeft'})
T16Left= Clean_data(T16Left)
T16Left= T16Left.rename(columns={'FAVORABLE':'Left'})
T16Center= Clean_data(T16Center)
T16Center= T16Center.rename(columns={'FAVORABLE':'Center'})
T16Right = Clean_data(T16Right)
T16Right= T16Right.rename(columns={'FAVORABLE':'Right'})
T16IPolls= Clean_data(T16IPolls)
T16IPolls= T16IPolls.rename(columns={'FAVORABLE':'Polls'})
T16UPolls= Clean_data(T16UPolls)
T16UPolls= T16UPolls.rename(columns={'FAVORABLE':'Uni Polls'})
'''
plt.plot(T16FarLeft['DAYS_TO_GENERAL_ELECTION'],T16FarLeft['FAVORABLE'], 'bd')
plt.plot(T16Left['DAYS_TO_GENERAL_ELECTION'],T16Left['FAVORABLE'], 'cd')
plt.plot(T16Center['DAYS_TO_GENERAL_ELECTION'],T16Center['FAVORABLE'],'md')
plt.plot(T16Right['DAYS_TO_GENERAL_ELECTION'],T16Right['FAVORABLE'],'rd')
plt.plot(T16IPolls['DAYS_TO_GENERAL_ELECTION'],T16IPolls['FAVORABLE'],'C1d')
plt.plot(T16UPolls['DAYS_TO_GENERAL_ELECTION'],T16UPolls['FAVORABLE'],'kd')
#plt.plot(NBC16['DAYS_TO_GENERAL_ELECTION'],NBC16['FAVORABLE'],'ro')
plt.legend()
'''

```




    "\nplt.plot(T16FarLeft['DAYS_TO_GENERAL_ELECTION'],T16FarLeft['FAVORABLE'], 'bd')\nplt.plot(T16Left['DAYS_TO_GENERAL_ELECTION'],T16Left['FAVORABLE'], 'cd')\nplt.plot(T16Center['DAYS_TO_GENERAL_ELECTION'],T16Center['FAVORABLE'],'md')\nplt.plot(T16Right['DAYS_TO_GENERAL_ELECTION'],T16Right['FAVORABLE'],'rd')\nplt.plot(T16IPolls['DAYS_TO_GENERAL_ELECTION'],T16IPolls['FAVORABLE'],'C1d')\nplt.plot(T16UPolls['DAYS_TO_GENERAL_ELECTION'],T16UPolls['FAVORABLE'],'kd')\n#plt.plot(NBC16['DAYS_TO_GENERAL_ELECTION'],NBC16['FAVORABLE'],'ro')\nplt.legend()\n"




```python
T16data = [T16FarLeft['FarLeft'],T16Left['Left'],T16Center['Center'],T16Right['Right'],T16IPolls['Polls'],T16UPolls['Uni Polls']]

T16headers = ['Left','Lean Left','Center','Lean Right','Ind Polls','Uni Polls',]

dfT16 = pd.concat(T16data, axis=1, keys=T16headers)

dfT16.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3cb48748>




![png](output_8_1.png)



```python
#Trump '20 polls
T20FarLeft = Trump20.loc[(Trump20['SOURCE']=='CNN/Opinion Research')|(Trump20['SOURCE']=='CNN/ORC')]
T20Left= Trump20.loc[(Trump20['SOURCE']=='Politico/Morning Consult')|(Trump20['SOURCE']=='NBC/WSJ')|(Trump20['SOURCE']=='ABC/Wash Post')|(Trump20['SOURCE']=='CBS News')|(Trump20['SOURCE']=='CBS News/NY Times')|(Trump20['SOURCE']=='Economist/YouGov')|(Trump20['SOURCE']=='The Atlantic/PRRI')]
T20Center = Trump20.loc[(Trump20['SOURCE']=='Associated Press/Gfk')|(Trump20['SOURCE']=='Bloomberg')|(Trump20['SOURCE']=='USA Today/Suffolk')]
T20Right = Trump20.loc[Trump20['SOURCE']=='FOX News']
#T20FarRight = Trump16.loc[Trump16['SOURCE']==']
T20IPolls = Trump20.loc[(Trump20['SOURCE']=='Gallup')|(Trump20['SOURCE']=='Gravis')|(Trump20['SOURCE']=='IDB/TIPP')|(Trump20['SOURCE']=='PPP (D)')|(Trump20['SOURCE']=='Rasmussen Resports')|(Trump20['SOURCE']=='Reuters/Ipsos')]
T20UPolls = Trump20.loc[(Trump20['SOURCE']=='Grinell/Selzer')|(Trump20['SOURCE']=='McClatchy/Marist')|(Trump20['SOURCE']=='GWU/Battlegroud')|(Trump20['SOURCE']=='Harvard-Harris')|(Trump20['SOURCE']=='GU Politics/Battleground')|(Trump20['SOURCE']=='Monmouth')|(Trump20['SOURCE']=='Quinnipiac')]

T20FarLeft = Clean_data(T20FarLeft)
T20FarLeft= T20FarLeft.rename(columns={'FAVORABLE':'FarLeft'})
T20Left= Clean_data(T20Left)
T20Left= T20Left.rename(columns={'FAVORABLE':'Left'})
T20Center= Clean_data(T20Center)
T20Center= T20Center.rename(columns={'FAVORABLE':'Center'})
T20Right = Clean_data(T20Right)
T20Right= T20Right.rename(columns={'FAVORABLE':'Right'})
T20IPolls= Clean_data(T20IPolls)
T20IPolls= T20IPolls.rename(columns={'FAVORABLE':'Polls'})
T20UPolls= Clean_data(T20UPolls)
T20UPolls= T20UPolls.rename(columns={'FAVORABLE':'Uni Polls'})

'''
plt.plot(T20FarLeft['DAYS_TO_GENERAL_ELECTION'],T20FarLeft['FAVORABLE'], 'bd')
plt.plot(T20Left['DAYS_TO_GENERAL_ELECTION'],T20Left['FAVORABLE'], 'cd')
plt.plot(T20Center['DAYS_TO_GENERAL_ELECTION'],T20Center['FAVORABLE'],'md')
plt.plot(T20Right['DAYS_TO_GENERAL_ELECTION'],T20Right['FAVORABLE'],'rd')
plt.plot(T20IPolls['DAYS_TO_GENERAL_ELECTION'],T20IPolls['FAVORABLE'],'C1d')
plt.plot(T20UPolls['DAYS_TO_GENERAL_ELECTION'],T20UPolls['FAVORABLE'],'kd')
plt.legend()
'''
```




    "\nplt.plot(T20FarLeft['DAYS_TO_GENERAL_ELECTION'],T20FarLeft['FAVORABLE'], 'bd')\nplt.plot(T20Left['DAYS_TO_GENERAL_ELECTION'],T20Left['FAVORABLE'], 'cd')\nplt.plot(T20Center['DAYS_TO_GENERAL_ELECTION'],T20Center['FAVORABLE'],'md')\nplt.plot(T20Right['DAYS_TO_GENERAL_ELECTION'],T20Right['FAVORABLE'],'rd')\nplt.plot(T20IPolls['DAYS_TO_GENERAL_ELECTION'],T20IPolls['FAVORABLE'],'C1d')\nplt.plot(T20UPolls['DAYS_TO_GENERAL_ELECTION'],T20UPolls['FAVORABLE'],'kd')\nplt.legend()\n"




```python
T20data = [T20FarLeft['FarLeft'],T20Left['Left'],T20Center['Center'],T20Right['Right'],T20IPolls['Polls'],T20UPolls['Uni Polls']]

T20headers = ['Left','Lean Left','Center','Lean Right','Ind Polls','Uni Polls',]

dfT20 = pd.concat(T20data, axis=1, keys=T20headers)

dfT20.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3cc48588>




![png](output_10_1.png)



```python
#Bernie'16 polls
B16FarLeft = Bernie16.loc[(Bernie16['SOURCE']=='CNN/ORC')]
B16Left= Bernie16.loc[(Bernie16['SOURCE']=='NBC/WSJ')|(Bernie16['SOURCE']=='ABC/Wash Post')|(Bernie16['SOURCE']=='CBS News')|(Bernie16['SOURCE']=='CBS News/NY Times')|(Bernie16['SOURCE']=='Economist/YouGov')|(Bernie16['SOURCE']=='The Atlantic/PRRI')]
B16Center = Bernie16.loc[(Bernie16['SOURCE']=='Associated Press/Gfk')|(Bernie16['SOURCE']=='Bloomberg')|(Bernie16['SOURCE']=='USA Today/Suffolk')]
B16Right = Bernie16.loc[Bernie16['SOURCE']=='FOX News']
#B16FarRight = Bernie16.loc[Bernie16['SOURCE']==']
B16IPolls = Bernie16.loc[(Bernie16['SOURCE']=='Gallup')|(Bernie16['SOURCE']=='PPP (D)')|(Bernie16['SOURCE']=='Rasmussen Resports')]
B16UPolls = Bernie16.loc[(Bernie16['SOURCE']=='McClatchy/Marist')|(Bernie16['SOURCE']=='GWU/Battlegroud')|(Bernie16['SOURCE']=='Monmouth')|(Bernie16['SOURCE']=='Quinnipiac')]

plt.plot(B16FarLeft['DAYS_TO_GENERAL_ELECTION'],B16FarLeft['FAVORABLE'], 'bd')
plt.plot(B16Left['DAYS_TO_GENERAL_ELECTION'],B16Left['FAVORABLE'], 'cd')
plt.plot(B16Center['DAYS_TO_GENERAL_ELECTION'],B16Center['FAVORABLE'],'md')
plt.plot(B16Right['DAYS_TO_GENERAL_ELECTION'],B16Right['FAVORABLE'],'rd')
plt.plot(B16IPolls['DAYS_TO_GENERAL_ELECTION'],B16IPolls['FAVORABLE'],'C1d')
plt.plot(B16UPolls['DAYS_TO_GENERAL_ELECTION'],B16UPolls['FAVORABLE'],'kd')
plt.legend()

```




    <matplotlib.legend.Legend at 0x1c30de4da0>




![png](output_11_1.png)



```python
#Bernie'20 polls
#B20FarLeft
B20Left= Bernie20.loc[(Bernie20['SOURCE']=='NBC/WSJ')|(Bernie20['SOURCE']=='Economist/YouGov')|(Bernie20['SOURCE']=='CNN')|(Bernie20['SOURCE']=='NPR/PBS/Marist')|(Bernie20['SOURCE']=='Politico/Morning Consult')]
#B20Center = Bernie20.loc[(Bernie20
B20Right = Bernie20.loc[Bernie20['SOURCE']=='FOX News']
#B20FarRight = Bernie20.loc[Bernie20['SOURCE']==']
#B20IPolls = 
B20UPolls = Bernie20.loc[(Bernie20['SOURCE']=='GU Politics/Battlegroud')|(Bernie20['SOURCE']=='Monmouth')|(Bernie20['SOURCE']=='Quinnipiac')|(Bernie20['SOURCE']=='Grinnell/Selzer')|(Bernie20['SOURCE']=='Harvard-Harris')]

plt.plot(B20Left['DAYS_TO_GENERAL_ELECTION'],B20Left['FAVORABLE'], 'cd')
plt.plot(B20Right['DAYS_TO_GENERAL_ELECTION'],B20Right['FAVORABLE'],'rd')
plt.plot(B20UPolls['DAYS_TO_GENERAL_ELECTION'],B20UPolls['FAVORABLE'],'kd')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a16d34f60>




![png](output_12_1.png)



```python
m2, b2, c2, d2 = np.polyfit(B20Left['DAYS_TO_GENERAL_ELECTION'],B20Left['FAVORABLE'],3)

plt.plot(B20Left['DAYS_TO_GENERAL_ELECTION'],m2*(B20Left['DAYS_TO_GENERAL_ELECTION'])**3+b2*(B20Left['DAYS_TO_GENERAL_ELECTION'])**2+c2*(B20Left['DAYS_TO_GENERAL_ELECTION'])+d2, 'co')

m4, b4 = np.polyfit(B20Right['DAYS_TO_GENERAL_ELECTION'],B20Right['FAVORABLE'],1)

plt.plot(B20Right['DAYS_TO_GENERAL_ELECTION'],m4*B20Right['DAYS_TO_GENERAL_ELECTION']+b4, 'ro')

m5, b5, c5, d5 = np.polyfit(B20UPolls['DAYS_TO_GENERAL_ELECTION'],B20UPolls['FAVORABLE'],3)

plt.plot(B20UPolls['DAYS_TO_GENERAL_ELECTION'],m5*(B20UPolls['DAYS_TO_GENERAL_ELECTION'])**3+b5*(B20UPolls['DAYS_TO_GENERAL_ELECTION'])**2+c5*(B20UPolls['DAYS_TO_GENERAL_ELECTION'])+d5, 'ko')

```




    [<matplotlib.lines.Line2D at 0x1a22681f60>]




![png](output_13_1.png)



```python
#Biden'20 polls
#BFarLeft
BLeft= Biden.loc[(Biden['SOURCE']=='NBC/WSJ')|(Biden['SOURCE']=='Economist/YouGov')|(Biden['SOURCE']=='CNN')|(Biden['SOURCE']=='NPR/PBS/Marist')|(Biden['SOURCE']=='Politico/Morning Consult')]
#BCenter = Biden.loc[(Biden
BRight = Biden.loc[Biden['SOURCE']=='FOX News']
#BFarRight = Biden.loc[Biden['SOURCE']==']
#BIPolls = 
BUPolls = Biden.loc[(Biden['SOURCE']=='GU Politics/Battlegroud')|(Biden['SOURCE']=='Monmouth')|(Biden['SOURCE']=='Quinnipiac')|(Biden['SOURCE']=='Grinnell/Selzer')|(Biden['SOURCE']=='Harvard-Harris')]

BLeft= Clean_data(BLeft)
BLeft= BLeft.rename(columns={'FAVORABLE':'BLeft'})
BRight = Clean_data(BRight)
BRight= BRight.rename(columns={'FAVORABLE':'BRight'})
BUPolls= Clean_data(BUPolls)
BUPolls= BUPolls.rename(columns={'FAVORABLE':'BUPolls'})

'''
#plt.plot(BFarLeft['DAYS_TO_GENERAL_ELECTION'],BFarLeft['FAVORABLE'], 'bd')
plt.plot(BLeft['DAYS_TO_GENERAL_ELECTION'],BLeft['FAVORABLE'], 'cd')
#plt.plot(BCenter['DAYS_TO_GENERAL_ELECTION'],BCenter['FAVORABLE'],'md')
plt.plot(BRight['DAYS_TO_GENERAL_ELECTION'],BRight['FAVORABLE'],'rd')
#plt.plot(BIPolls['DAYS_TO_GENERAL_ELECTION'],BIPolls['FAVORABLE'],'C1d')
plt.plot(BUPolls['DAYS_TO_GENERAL_ELECTION'],BUPolls['FAVORABLE'],'kd')
plt.legend()
print(BLeft)
'''
```




    "\n#plt.plot(BFarLeft['DAYS_TO_GENERAL_ELECTION'],BFarLeft['FAVORABLE'], 'bd')\nplt.plot(BLeft['DAYS_TO_GENERAL_ELECTION'],BLeft['FAVORABLE'], 'cd')\n#plt.plot(BCenter['DAYS_TO_GENERAL_ELECTION'],BCenter['FAVORABLE'],'md')\nplt.plot(BRight['DAYS_TO_GENERAL_ELECTION'],BRight['FAVORABLE'],'rd')\n#plt.plot(BIPolls['DAYS_TO_GENERAL_ELECTION'],BIPolls['FAVORABLE'],'C1d')\nplt.plot(BUPolls['DAYS_TO_GENERAL_ELECTION'],BUPolls['FAVORABLE'],'kd')\nplt.legend()\nprint(BLeft)\n"




```python
Bdata = [BLeft['BLeft'],BRight['BRight'],BUPolls['BUPolls']]

Bheaders = ['Left','Lean Right','Uni Polls',]

dfB = pd.concat(Bdata, axis=1, keys=Bheaders)

dfB.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3ca07208>




![png](output_15_1.png)



```python
XH= Trump16[['DAYS_TO_GENERAL_ELECTION','MARGIN']]
yH = Trump16['FAVORABLE']
X_train, X_test, y_train, y_test = train_test_split(XH, yH, test_size = .9, random_state=42) 
```


```python
lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit

y_test_preds = lm_model.predict(X_test)
"The r-squared score for the model using only quantitative variables was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))
```




    'The r-squared score for the model using only quantitative variables was 0.8321849922121847 on 192 values.'




```python
plt.plot(X_test['DAYS_TO_GENERAL_ELECTION'],y_test_preds,'o')

```




    [<matplotlib.lines.Line2D at 0x1a38747cc0>]




![png](output_18_1.png)



```python
B16FarLeft.head()
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
      <th>SOURCE</th>
      <th>SURVERYPERIOD</th>
      <th>DATEFORMATTED</th>
      <th>SAMPLE</th>
      <th>FAVORABLE</th>
      <th>UNFAVORABLE</th>
      <th>MARGIN</th>
      <th>CANDIDATE</th>
      <th>DAYS_TO_GENERAL_ELECTION</th>
      <th>ELECTION_YEAR</th>
      <th>ELECTION_DATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>158</th>
      <td>CNN/ORC</td>
      <td>2/24 - 2/27</td>
      <td>2/27/2016</td>
      <td>920 RV</td>
      <td>57</td>
      <td>33</td>
      <td>24</td>
      <td>BERNIE</td>
      <td>255</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>163</th>
      <td>CNN/ORC</td>
      <td>3/17 - 3/20</td>
      <td>3/20/2016</td>
      <td>1001 A</td>
      <td>48</td>
      <td>43</td>
      <td>5</td>
      <td>BERNIE</td>
      <td>233</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>166</th>
      <td>CNN/ORC</td>
      <td>4/28 - 5/1</td>
      <td>5/1/2016</td>
      <td>1001 A</td>
      <td>60</td>
      <td>32</td>
      <td>28</td>
      <td>BERNIE</td>
      <td>191</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
    <tr>
      <th>169</th>
      <td>CNN/ORC</td>
      <td>6/16 - 6/19</td>
      <td>6/19/2016</td>
      <td>891 RV</td>
      <td>59</td>
      <td>35</td>
      <td>24</td>
      <td>BERNIE</td>
      <td>142</td>
      <td>2016</td>
      <td>11/8/2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
import os
filepath = os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/Hilary.csv')
Hilary.to_csv(filepath,index=False)

```


```python
'''
T20FarLeft.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T20FarLeft.csv'),index=False)
T20Left.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T20Left.csv'),index=False)
T20Center.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T20Center'),index=False)
T20Right.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T20Right.csv'),index=False)
T20IPolls.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T20IPoll.csv'),index=False)
T20UPolls.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T20UPoll.csv'),index=False)

T16FarLeft.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T16FarLeft.csv'),index=False)
T16Left.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T16Left.csv'),index=False)
T16Center.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T16Center'),index=False)
T16Right.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T16Right.csv'),index=False)
T16IPolls.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T16IPoll.csv'),index=False)
T16UPolls.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/T16UPoll.csv'),index=False)

HFarLeft.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/HFarLeft.csv'),index=False)
HLeft.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/HLeft.csv'),index=False)
HCenter.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/HCenter'),index=False)
HRight.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/HRight.csv'),index=False)
HIPolls.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/HIPoll.csv'),index=False)
HUPolls.to_csv(os.path.abspath('/Users/hannahknight/Data Science/Data Science Project 1 Blog/HUPoll.csv'),index=False)
'''
```


```python
fig, ax = plt.subplots(figsize=(5,3),dpi=200)
plt.title("Hillary Polls")
dfH.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3d25b0f0>




![png](output_22_1.png)



```python
fig, ax = plt.subplots(figsize=(5,3),dpi=200)
plt.title("Trump '16 Polls")
dfT16.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3d328400>




![png](output_23_1.png)



```python
fig, ax = plt.subplots(figsize=(5,3),dpi=200)
plt.title("Trump '20 Polls")
dfT20.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3d604710>




![png](output_24_1.png)



```python
fig, ax = plt.subplots(figsize=(5,3),dpi=200)
plt.title("Biden Polls")
dfB.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c3cb59588>




![png](output_25_1.png)

