#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns
%matplotlib inline
sns.set(style='ticks',color_codes=True)
from pylab import rcParams
rcParams['figure.figsize'] = 15,10
warnings.simplefilter('ignore')
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import train_test_split,KFold
from yellowbrick.features import FeatureImportances
from yellowbrick.target import BalancedBinningReference
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot,AlphaSelection
from yellowbrick.model_selection import CVScores,LearningCurve
# %%
data = pd.read_csv('concrete.csv')
data.head()
data.describe()
# %%
features = ['cement','slag','ash','water','splast','coarse','age']
target = 'strength'
X = data[features]
y = data[target]
# %%
sns.pairplot(data)
# %%
fig = plt.figure()
ax = fig.add_subplot()
labels = list(map(lambda s: s.title(),features))
viz = FeatureImportances(Lasso(),ax=ax,labels=labels,relative=False)
viz.fit(X,y)
viz.poof()
# %%
from yellowbrick.target import BalancedBinningReference
visualizer = BalancedBinningReference()
visualizer.fit(y)
visualizer.poof()
# %%
from yellowbrick.regressor import PredictionError
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
visualizer = PredictionError(Lasso(),size= (800,600))
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
visualizer.finalize()
visualizer.ax.set_xlabel('Measured Concrete Strength')
visualizer.ax.set_ylabel('Predicted Concrete strength')
# %%
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(Lasso(),size = (800,600))
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
g = visualizer.poof()
# %%
from sklearn.model_selection import KFold
from yellowbrick.model_selection import CVScores
_,ax = plt.subplots()
cv = KFold(12)
oz = CVScores(Lasso(),ax=ax,scoring='r2')
oz.fit(X_train,y_train)
oz.poof()
# %%
from yellowbrick.model_selection import LearningCurve
from sklearn.linear_model import LassoCV
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
sizes = np.linspace(0.3,1.0,10)
viz = LearningCurve(LassoCV(),train_sizes=sizes,scoring='r2')
viz.fit(X,y)
viz.poof()
# %%
from sklearn.linear_model import LassoCV
from yellowbrick.regressor import AlphaSelection
alphas = np.logspace(-10,1,400)
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model,size = (800,600))
visualizer.fit(X,y)
visualizer.poof()