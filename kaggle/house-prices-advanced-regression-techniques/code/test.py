import pandas as pd
import numpy as np

#'Neighborhood',

def read_data(filename):
    data = pd.read_csv(filename, index_col='Id',
                       usecols=['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
                                'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                                'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
                                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
                                'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                                'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
                                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
                                'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                                'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
                                'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
                                'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
                                'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                                'ScreenPorch', 'PoolArea', 'MiscVal',
                                'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'],
                       converters={'MSZoning': str
                           , 'Street': str
                           , 'Alley': str
                           , 'LotShape': str
                           , 'LandContour': str
                           , 'Utilities': str
                           , 'LotConfig': str
                           , 'LandSlope': str
                           , 'Condition1': str
                           , 'Condition2': str
                           , 'BldgType': str
                           , 'HouseStyle': str
                           , 'RoofStyle': str
                           , 'RoofMatl': str
                           , 'Exterior1st': str
                           , 'Exterior2nd': str
                           , 'MasVnrType': str
                           , 'ExterQual': str
                           , 'ExterCond': str
                           , 'Foundation': str
                           , 'BsmtQual': str
                           , 'BsmtCond': str
                           , 'BsmtExposure': str
                           , 'BsmtFinType1': str
                           , 'BsmtFinType2': str
                           , 'Heating': str
                           , 'HeatingQC': str
                           , 'CentralAir': str
                           , 'Electrical': str
                           , 'KitchenQual': str
                           , 'Functional': str
                           , 'FireplaceQu': str
                           , 'GarageType': str
                           , 'GarageFinish': str
                           , 'GarageQual': str
                           , 'GarageCond': str
                           , 'PavedDrive': str
                           , 'SaleType': str
                           , 'SaleCondition': str})

    need_fills = ['MasVnrArea', 'LotFrontage', 'GarageYrBlt']
    for col in need_fills:
        data[col] = data[col].fillna(data[col].mean())

    data['SalePrice'] = data['SalePrice'].astype(float)
    # data=data.dropna()
    # print(len(data))
    # exit()

    maps = {
        'MSZoning': {
            'A': 0,
            'C': 1,
            'C (all)': 1,
            'FV': 2,
            'I': 3,
            'RH': 4,
            'RL': 5,
            'RP': 6,
            'RM': 7
        },
        'Street': {
            'NA': 0,
            'Grvl': 1,
            'Pave': 2
        },
        'Alley': {
            'NA': 0,
            'Grvl': 1,
            'Pave': 2
        },
        'LotShape': {
            'Reg': 1,
            'IR1': 2,
            'IR2': 3,
            'IR3': 4,
        },
        'LandContour': {
            'Lvl': 4,
            'Bnk': 3,
            'HLS': 2,
            'Low': 1
        },
        'Utilities': {
            'AllPub': 4,
            'NoSewr': 3,
            'NoSeWa': 2,
            'ELO': 1
        },
        'LotConfig': {
            'Inside': 5,
            'Corner': 4,
            'CulDSac': 3,
            'FR2': 2,
            'FR3': 1
        },
        'LandSlope': {
            'Gtl': 3,
            'Mod': 2,
            'Sev': 1
        },
        'Condition1': {
            'Artery': 9,
            'Feedr': 8,
            'Norm': 7,
            'RRNn': 6,
            'RRAn': 5,
            'PosN': 4,
            'PosA': 3,
            'RRNe': 2,
            'RRAe': 1
        },
        'Condition2': {
            'Artery': 9,
            'Feedr': 8,
            'Norm': 7,
            'RRNn': 6,
            'RRAn': 5,
            'PosN': 4,
            'PosA': 3,
            'RRNe': 2,
            'RRAe': 1
        }, 'BldgType': {
            '1Fam': 1,
            '2FmCon': 2,
            '2fmCon': 2,
            'Duplx': 3,
            'Duplex': 3,
            'TwnhsE': 4,
            'TwnhsI': 5,
            'Twnhs': 4
        }
        , 'HouseStyle': {
            '1Story': 1,
            '1.5Fin': 2,
            '1.5Unf': 3,
            '2Story': 4,
            '2.5Fin': 5,
            '2.5Unf': 6,
            'SFoyer': 7,
            'SLvl': 8,
        }
        , 'RoofStyle': {
            'Flat': 1,
            'Gable': 2,
            'Gambrel': 3,
            'Hip': 4,
            'Mansard': 5,
            'Shed': 6,
        }
        , 'RoofMatl': {
            'ClyTile': 1,
            'CompShg': 2,
            'Membran': 3,
            'Metal': 4,
            'Roll': 5,
            'Tar&Grv': 6,
            'WdShake': 7,
            'WdShngl': 8,
        }
        , 'Exterior1st': {
            'AsbShng': 1,
            'AsphShn': 2,
            'BrkComm': 3,
            'BrkFace': 4,
            'CBlock': 5,
            'CemntBd': 6,
            'HdBoard': 7,
            'ImStucc': 8,
            'MetalSd': 9,
            'Other': 10,
            'Plywood': 11,
            'PreCast': 12,
            'Stone': 13,
            'Stucco': 14,
            'VinylSd': 15,
            'Wd Sdng': 16,
            'WdShing': 17,
        }
        , 'Exterior2nd': {
            'AsbShng': 1,
            'AsphShn': 2,
            'BrkComm': 3,
            'Brk Cmn': 3,
            'BrkFace': 4,
            'CBlock': 5,
            'CemntBd': 6,
            'CmentBd': 6,
            'HdBoard': 7,
            'ImStucc': 8,
            'MetalSd': 9,
            'Other': 10,
            'Plywood': 11,
            'PreCast': 12,
            'Stone': 13,
            'Stucco': 14,
            'VinylSd': 15,
            'Wd Sdng': 16,
            'WdShing': 17,
            'Wd Shng': 17,
        }
        , 'MasVnrType': {
            'BrkCmn': 1,
            'BrkFace': 2,
            'CBlock': 3,
            'Stone': 4,
            'None': 5,
            'NA': 5
        }
        , 'ExterQual': {
            'Ex': 1,
            'Gd': 2,
            'TA': 3,
            'Fa': 4,
            'Po': 5,
        }
        , 'ExterCond': {
            'Ex': 1,
            'Gd': 2,
            'TA': 3,
            'Fa': 4,
            'Po': 5,
        }
        , 'Foundation': {
            'BrkTil': 1,
            'CBlock': 2,
            'PConc': 3,
            'Slab': 4,
            'Stone': 5,
            'Wood': 6,
        }
        , 'BsmtQual': {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 0,
        }
        , 'BsmtCond': {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 0,
        }
        , 'BsmtExposure': {
            'Gd': 4,
            'Av': 3,
            'Mn': 2,
            'No': 1,
            'NA': 0,
        }
        , 'BsmtFinType1': {
            'GLQ': 6,
            'ALQ': 5,
            'BLQ': 4,
            'Rec': 3,
            'LwQ': 2,
            'Unf': 1,
            'NA': 0,
        }
        , 'BsmtFinType2': {
            'GLQ': 6,
            'ALQ': 5,
            'BLQ': 4,
            'Rec': 3,
            'LwQ': 2,
            'Unf': 1,
            'NA': 0,
        }
        , 'Heating': {
            'Floor': 1,
            'GasA': 2,
            'GasW': 3,
            'Grav': 4,
            'OthW': 5,
            'Wall': 6,
        }
        , 'HeatingQC': {
            'Ex': 1,
            'Gd': 2,
            'TA': 3,
            'Fa': 4,
            'Po': 5,
        }
        , 'CentralAir': {
            'N': 0,
            'Y': 1,
        }
        , 'Electrical': {
            'SBrkr': 1,
            'FuseA': 2,
            'FuseF': 3,
            'FuseP': 4,
            'Mix': 5,
            'NA': 0
        }
        , 'KitchenQual': {
            'Ex': 1,
            'Gd': 2,
            'TA': 3,
            'Fa': 4,
            'Po': 5,
        }
        , 'Functional': {
            'Typ': 1,
            'Min1': 2,
            'Min2': 3,
            'Mod': 4,
            'Maj1': 5,
            'Maj2': 6,
            'Sev': 7,
            'Sal': 8,
        }
        , 'FireplaceQu': {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 0,
        }
        , 'GarageType': {
            '2Types': 6,
            'Attchd': 5,
            'Basment': 4,
            'BuiltIn': 3,
            'CarPort': 2,
            'Detchd': 1,
            'NA': 0,
        }
        , 'GarageFinish': {
            'Fin': 3,
            'RFn': 2,
            'Unf': 1,
            'NA': 0,
        }
        , 'GarageQual': {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 0,
        }
        , 'GarageCond': {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 0,
        }
        , 'PavedDrive': {
            'Y': 1,
            'P': 2,
            'N': 3,
        }
        , 'PoolQC': {
            'Ex': 4,
            'Gd': 3,
            'TA': 2,
            'Fa': 1,
            'NA': 0,
        }
        , 'Fence': {
            'GdPrv': 4,
            'MnPrv': 3,
            'GdWo': 2,
            'MnWw': 1,
            'NA': 0,
        }
        , 'MiscFeature': {
            'Elev': 1,
            'Gar2': 2,
            'Othr': 3,
            'Shed': 4,
            'TenC': 5,
            'NA': 6,
        }
        , 'SaleType': {
            'WD': 1,
            'CWD': 2,
            'VWD': 3,
            'New': 4,
            'COD': 5,
            'Con': 6,
            'ConLw': 7,
            'ConLI': 8,
            'ConLD': 9,
            'Oth': 10,
        }
        , 'SaleCondition': {
            'Normal': 1,
            'Abnorml': 2,
            'AdjLand': 3,
            'Alloca': 4,
            'Family': 5,
            'Partial': 6,
        }

    }

    for a in maps:
        if a in data.columns:
            data[a] = data[a].map(maps[a])
    return data

data=read_data('../data/train.csv')
#
# d=data.isnull().sum()
# print(d[d>0])
# for i in d[d>0].index:
#     print("%s:" % i)
#     print(np.unique(data[i]))
# exit()

# for col in data.columns:
#     if data[col].dtype!=np.float and data[col].dtype!=np.int:
#         values=data[col].unique()
#         map1={}
#         for i in range(0,len(values)):
#             map1[values[i]]=i
#         data[col]=data[col].map(map1)
#         print('enter')

from random import shuffle

idx=data.index.values
shuffle(idx) #随机打乱数据
data.reindex(idx)
p = 0.8 #设置训练数据比例
train = data.iloc[:int(len(data)*p),:] #前80%为训练集
test = data.iloc[int(len(data)*p):,:] #后20%为测试集



from LassoModel import LassoModel

model = LassoModel(train)
model.glmnet(train)
# model.fit(train)
# model.gray(train)
model.deep_learn_train(train,test)
r=model.predict_data(test)
r=r.reshape((1,-1))[0]
# r=model.lr(train.iloc[:,:-1][model.columns],train.iloc[:,-1],test.iloc[:,:-1][model.columns])
print(r)

import matplotlib.pyplot as plt
def plot_err(r,c):
    result = (r - test.iloc[:, -1]) / test.iloc[:, -1]
    result = result.sort_index()
    result.plot(color=c)


plot_err(r,'r')

train=train[model.columns]
test=test[model.columns]
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_jobs=4,n_estimators=100,criterion='mse',random_state=1,oob_score=True)
rf.fit(train.iloc[:,:-1],train.iloc[:,-1])
r=rf.predict(test.iloc[:,:-1])
plot_err(r,'b')

plt.show()


from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(data)
# print(pca.components_) #返回模型的各个特征向量
# print(pca.explained_variance_ratio_) #返回各个成分各自的方差百分比
# low_d=pca.transform(data)
# print(low_d)
# exit()