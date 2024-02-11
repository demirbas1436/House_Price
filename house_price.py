import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import warnings
warnings.simplefilter(action="ignore")

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
################################
# House Price Prediction Model
################################
##################################################################################################################
# Task
# Develop a machine learning model that predicts house prices with minimum error based on the data set we have.
##################################################################################################################
############################################################################
# TASK 1 : Apply EDA processes to the data set.
#
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)
############################################################################
df = pd.concat([train, test], ignore_index=False).reset_index()
df = df.drop("index", axis=1)
#print(df.shape) #(1460, 81)   (1459, 80)
#fark = set(df.columns) - set(train.columns)
#print(fark)

########################################################################################################################
# Id: Evlerin benzersiz kimlik numarasıdır.
# MSSubClass: Evlerin yapı sınıfını gösterir.
#             Örneğin, 20 = 1 katlı, 60 = 2 katlı, 120 = 1 katlı townhouse.
# MSZoning: Evlerin genel bölgeleme sınıfını gösterir.
#           Örneğin, RL = Düşük yoğunluklu konut, RM = Orta yoğunluklu konut, C = Ticari.
# LotFrontage: Evlerin caddeye bakan cephesinin uzunluğudur.
#              Metre cinsinden ifade edilir.
# LotArea: Evlerin arsa alanıdır. Metrekare cinsinden ifade edilir.
# Street: Evlerin cadde erişim tipini gösterir.
#         Örneğin, Pave = Asfalt, Grvl = Çakıl gibi.
# Alley: Evlerin arka sokak erişim tipini gösterir.
#        Örneğin, Pave = Asfalt, Grvl = Çakıl, NA = Yok.
# LotShape: Evlerin arsa şeklini gösterir.
#           Örneğin, Reg = Düzenli, IR1 = Hafif eğri, IR2 = Orta eğri, IR3 = Güçlü eğri.
# LandContour: Evlerin arsa düzlüğünü gösterir.
#              Örneğin, Lvl = Düz, Bnk = Banketli, HLS = Tepe, Low = Alçak.
# Utilities: Evlerin kullanılabilir hizmetlerini gösterir.
#            Örneğin, AllPub=Tüm kamu hizmetleri, NoSewr=Kanalizasyon hariç, NoSeWa=Elektrik ve gaz hariç, ELO=Hiçbiri.
# LotConfig: Evlerin arsa konfigürasyonunu gösterir.
#            Örneğin, Inside = İç köşe, Corner = Dış köşe, CulDSac = Çıkmaz sokak, FR2 = İki cephe, FR3 = Üç cephe.
# LandSlope: Evlerin arsa eğimini gösterir.
#            Örneğin, Gtl = Hafif eğimli, Mod = Orta eğimli, Sev = Şiddetli eğimli.
# Neighborhood: Evlerin bulunduğu mahallenin adıdır.
#               Örneğin, CollgCr = College Creek, Veenker = Veenker, Crawfor = Crawford.
# Condition1: Evlerin yakınındaki ana yol veya demiryolu durumunu gösterir.
#             Örneğin, Norm = Normal, Feedr = Besleyici yol, PosN = Yakın olumlu özellik.
# Condition2: Evlerin yakınındaki ikincil yol veya demiryolu durumunu gösterir.
#             Örneğin, Norm = Normal, Artery = Ana yol, RRNn = Yakın kuzey demiryolu.
# BldgType: Evlerin yapı tipini gösterir.
#           Örneğin, 1Fam = Tek aileli, 2fmCon = İki aileli dönüşüm, Duplex = Dubleks.
# HouseStyle: Evlerin yaşam alanı stilini gösterir.
#             Örneğin, 1Story = 1 katlı, 2Story = 2 katlı, 1.5Fin = 1.5 katlı tamamlanmış.
# OverallQual: Evlerin genel malzeme ve bitiş kalitesini gösterir.
#              1-10 arasında bir sayıdır. 1 = Çok kötü, 10 = Çok iyi.
# OverallCond: Evlerin genel durumunu gösterir.
#              1-10 arasında bir sayıdır. 1 = Çok kötü, 10 = Çok iyi.
# YearBuilt: Evlerin inşa edildiği yıldır.
#            4 basamaklı bir sayıdır. Örneğin, 2003, 1976, 1925.
# YearRemodAdd: Evlerin yeniden modelleme veya ilave yapıldığı yıldır.
#               4 basamaklı bir sayıdır. Örneğin, 2004, 1976, 1950.
# RoofStyle: Evlerin çatı stilini gösterir.
#            Örneğin, Flat = Düz, Gable = Çatı, Gambrel = Kırma çatı.
# RoofMatl: Evlerin çatı malzemesini gösterir.
#           Örneğin, CompShg = Kompozit shingle, Tar&Grv = Katran ve çakıl, WdShake = Ahşap shake.
# Exterior1st: Evlerin dış kaplama malzemesini gösterir.
#              Örneğin, VinylSd = Vinil siding, HdBoard = Sert tahta, Plywood = Kontrplak.
# Exterior2nd: Evlerin ikincil dış kaplama malzemesini gösterir.
#              Örneğin, VinylSd = Vinil siding, HdBoard = Sert tahta, Plywood = Kontrplak.
# MasVnrType: Evlerin duvar kaplama tipini gösterir.
#             Örneğin, BrkFace = Tuğla yüzey, None = Yok, Stone = Taş.
# MasVnrArea: Evlerin duvar kaplama alanıdır. Metrekare cinsinden ifade edilir.
# ExterQual: Evlerin dış malzeme kalitesini gösterir.
#            Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü.
# ExterCond: Evlerin dış malzeme durumunu gösterir.
#            Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü.
# Foundation: Evlerin temel tipini gösterir.
#             Örneğin, PConc = Betonarme, CBlock = Beton blok, BrkTil = Tuğla kiremit.
# BsmtQual: Evlerin bodrum yüksekliğini gösterir.
#           Örneğin, Ex = 100+ inç, Gd = 90-99 inç, TA = 80-89 inç, Fa = 70-79 inç, Po = <70 inç, NA = Bodrum yok.
# BsmtCond: Evlerin bodrum durumunu gösterir.
#           Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü, NA = Bodrum yok.
# BsmtExposure: Evlerin bodrum cephesini gösterir.
#               Örneğin, Gd=İyi maruziyet, Av=Ortalama maruziyet, Mn=Minimum maruziyet, No=Maruziyet yok, NA=Bodrum yok.
# BsmtFinType1: Evlerin birinci bodrum bitiş kalitesini gösterir.
#               Örneğin, GLQ = İyi oturma odası, ALQ = Ortalama oturma odası, BLQ = Aşağıda ortalama oturma odası,
#               Rec = Oyun odası, LwQ = Düşük kaliteli oturma odası, Unf = Bitmemiş, NA = Bodrum yok.
# BsmtFinSF1: Evlerin birinci bodrum bitiş alanıdır. Metrekare cinsinden ifade edilir.
# BsmtFinType2: Evlerin ikinci bodrum bitiş kalitesini gösterir.
#               Örneğin, GLQ = İyi oturma odası, ALQ = Ortalama oturma odası, BLQ = Aşağıda ortalama oturma odası,
#               Rec = Oyun odası, LwQ = Düşük kaliteli oturma odası, Unf = Bitmemiş, NA = Bodrum yok.
# BsmtFinSF2: Evlerin ikinci bodrum bitiş alanıdır. Metrekare cinsinden ifade edilir.
# BsmtUnfSF: Evlerin bitmemiş bodrum alanıdır. Metrekare cinsinden ifade edilir.
# TotalBsmtSF: Evlerin toplam bodrum alanıdır. Metrekare cinsinden ifade edilir.
# Heating: Evlerin ısıtma tipini gösterir.
#          Örneğin, Floor = Zemin ısıtma, GasA = Gazlı hava, GasW = Gazlı su.
# HeatingQC: Evlerin ısıtma kalitesini ve durumunu gösterir.
#            Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü.
# CentralAir: Evlerin merkezi klima sistemine sahip olup olmadığını gösterir.
#             Örneğin, Y = Var, N = Yok.
# Electrical: Evlerin elektrik sisteminin tipini gösterir.
#             Örneğin, SBrkr = Standart devre kesici, FuseA = 60 amper sigorta kutusu, FuseF = 60 amper sigorta kutusu.
# 1stFlrSF: Evlerin birinci kat alanıdır. Metrekare cinsinden ifade edilir.
# 2ndFlrSF: Evlerin ikinci kat alanıdır. Metrekare cinsinden ifade edilir.
# LowQualFinSF: Evlerin düşük kaliteli bitiş alanıdır. Metrekare cinsinden ifade edilir.
# GrLivArea: Evlerin zemin üstü yaşam alanıdır. Metrekare cinsinden ifade edilir.
# BsmtFullBath: Evlerin bodrum katındaki tam banyo sayısıdır.
#               Tam banyo, küvet, lavabo ve tuvalet içeren banyodur.
# BsmtHalfBath: Evlerin bodrum katındaki yarım banyo sayısıdır.
#               Yarım banyo, sadece lavabo ve tuvalet içeren banyodur.
# FullBath: Evlerin zemin üstü katlarındaki tam banyo sayısıdır.
#           Tam banyo, küvet, lavabo ve tuvalet içeren banyodur.
# HalfBath: Evlerin zemin üstü katlarındaki yarım banyo sayısıdır.
#           Yarım banyo, sadece lavabo ve tuvalet içeren banyodur.
# BedroomAbvGr: Evlerin zemin üstü katlarındaki yatak odası sayısıdır.
# KitchenAbvGr: Evlerin zemin üstü katlarındaki mutfak sayısıdır.
# KitchenQual: Evlerin mutfak kalitesini gösterir.
#              Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü.
# TotRmsAbvGrd: Evlerin zemin üstü katlarındaki toplam oda sayısıdır. Banyolar dahil değildir.
# Functional: Evlerin işlevselliğini gösterir.
#             Örneğin, Typ = Tipik, Min1 = Minör hasar 1, Min2 = Minör hasar 2, Maj1 = Büyük hasar 1,
#             Maj2 = Büyük hasar 2, Sev = Ciddi hasar, Sal = Sadece yıkım.
# Fireplaces: Evlerin şömine sayısıdır.
# FireplaceQu: Evlerin şömine kalitesini gösterir.
#              Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü, NA = Şömine yok.
# GarageType: Evlerin garaj tipini gösterir.
#             Örneğin, Attchd = Bitişik, Detchd = Ayrık, BuiltIn = Gömme, CarPort = Araba portu, NA = Garaj yok.
# GarageYrBlt: Evlerin garajının inşa edildiği yıldır. 4 basamaklı bir sayıdır.
#              Örneğin, 2005, 1978, 1930.
# GarageFinish: Evlerin garajının bitiş durumunu gösterir.
#               Örneğin, Fin = Tamamlanmış, RFn = Kaba tamamlanmış, Unf = Bitmemiş, NA = Garaj yok.
# GarageCars: Evlerin garajındaki araba kapasitesidir. Bir sayıdır. Örneğin, 2, 3, 4 gibi.
# GarageArea: Evlerin garaj alanıdır. Metrekare cinsinden ifade edilir.
# GarageQual: Evlerin garaj kalitesini gösterir.
#             Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü, NA = Garaj yok.
# GarageCond: Evlerin garaj durumunu gösterir.
#             Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, Po = Kötü, NA = Garaj yok.
# PavedDrive: Evlerin garaj girişinin asfaltlanmış olup olmadığını gösterir.
#             Örneğin, Y = Asfaltlanmış, P = Kısmen asfaltlanmış, N = Asfaltlanmamış.
# WoodDeckSF: Evlerin ahşap güverte alanıdır. Metrekare cinsinden ifade edilir.
# OpenPorchSF: Evlerin açık sundurma alanıdır. Metrekare cinsinden ifade edilir.
# EnclosedPorch: Evlerin kapalı sundurma alanıdır. Metrekare cinsinden ifade edilir.
# 3SsnPorch: Evlerin üç mevsimlik sundurma alanıdır. Metrekare cinsinden ifade edilir.
# ScreenPorch: Evlerin sineklikli sundurma alanıdır. Metrekare cinsinden ifade edilir.
# PoolArea: Evlerin havuz alanıdır. Metrekare cinsinden ifade edilir.
# PoolQC: Evlerin havuz kalitesini gösterir.
#         Örneğin, Ex = Mükemmel, Gd = İyi, TA = Ortalama, Fa = Adil, NA = Havuz yok.
# Fence: Evlerin çit durumunu gösterir.
#        Örneğin, GdPrv = İyi gizlilik, MnPrv = Minimum gizlilik, GdWo = İyi ahşap, MnWw = Minimum ahşap, NA = Çit yok.
# MiscFeature: Evlerin diğer kategorilere girmeyen özelliklerini gösterir.
#              Örneğin, Elev = Asansör, Gar2 = İkinci garaj, Othr = Diğer, NA = Yok.
# MiscVal: Evlerin diğer özelliklerinin değeridir. Dolar cinsinden ifade edilir.
# MoSold: Evlerin satıldığı aydır.
#         1-12 arasında bir sayıdır. Örneğin, 1 = Ocak, 2 = Şubat, 12 = Aralık.
# YrSold: Evlerin satıldığı yıldır.
#         4 basamaklı bir sayıdır. Örneğin, 2008, 2009, 2010.
# SaleType: Evlerin satış tipini gösterir.
#           Örneğin, WD = Normal satış, CWD = Nakit satış, VWD = VA satış, New = Yeni inşaat, COD = Mahkeme satışı,
#           Con = Sözleşme satışı, ConLw = Düşük faizli sözleşme satışı, ConLI = Düşük gelirli sözleşme satışı,
#           ConLD = Uzun vadeli sözleşme satışı, Oth = Diğer satış.
# SaleCondition: Evlerin satış durumunu gösterir.
#                Örneğin, Normal = Normal, Abnorml = Anormal, AdjLand = Arsa ayarlaması, Alloca = Tahsis,
#                Family = Aile satışı, Partial = Kısmi satış.
# SalePrice: Evlerin satış fiyatıdır. Dolar cinsinden ifade edilir.
########################################################################################################################
#print(df.dtypes)
#print(df.isnull().sum())




#######################################
# ANALYSİS OF CATEGORICAL VARIABLES
#######################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols


cat_cols, cat_but_car, num_cols = grab_col_names(df)
#print(cat_cols)
#print(cat_but_car)
#print(num_cols)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


#for col in cat_cols:
#    cat_summary(df, col)




###################################
# ANALYSIS OF NUMERİCAL VARIABLES
###################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


#for col in num_cols:
#    num_summary(df, col, True)


################################
# ANALYSIS OF TARGET VARIABLE
################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


#for col in cat_cols:
#    target_summary_with_cat(df, "SalePrice", col)


##############################
# ANALYSIS OF CORRELATION
##############################

corr = df[num_cols].corr()
#print(corr)

#sns.set(rc={"figure.figsize": (12, 12)})
#sns.heatmap(corr, cmap="RdBu")
#plt.show()


######################################
# Görev 2 : Feature Engineering
######################################

######################################
# OUTLIER ANALYSIS
######################################
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


#for col in num_cols:
#    print(col, check_outlier(df, col))

#print(df[df["LotFrontage"] < df["LotFrontage"].quantile(0.10)])
#print(df.iloc[56])

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)


#for col in num_cols:
#    print(col, check_outlier(df, col))



######################################
# MISSING VALUE ANALYSIS
######################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]


for col in no_cols:
    df[col].fillna("No", inplace=True)


def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)

print(missing_values_table(df, True))


######################################
#  RARE ANALYSIS AND RARE ENCODER
######################################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


#rare_analyser(df, "SalePrice", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df


rare_encoder(df, 0.01)


df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]
df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])
#df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
#                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual",
#                      "GarageCond", "Fence"]].sum(axis=1) # 42


df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"] # 32
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2 # 56
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF # 93
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF # 156
df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF # 35
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea # 64
df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea # 57
df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea # 69
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea # 36
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF) # 73
df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"] # 61
df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt # 31
df["NEW_HouseAge"] = df.YrSold - df.YearBuilt # 73
df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd # 40
df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt # 17
df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd) # 30
df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt # 48


drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating",
             "PoolQC", "MiscFeature", "Neighborhood"]

df.drop(drop_list, axis=1, inplace=True)


######################################
# Label Encoding & One-Hot Encoding
######################################
cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)


##################################
# TASK 3: MODEL BUILDING
##################################
##########################################################
# SalePrice değişkeni boş olan değerler test verisidir.
##########################################################
train_df = df[df["SalePrice"].notnull()]
test_df = df[df["SalePrice"].isnull()]

y = train_df["SalePrice"] # np.log1p(df["SalePrice"])
X = train_df.drop(["Id", "SalePrice"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LinearRegression().fit(X, y)
y_pred = log_model.predict(X)
#print(mean_squared_error(y, y_pred)) #470189136.0439013
#print(y.mean()) #180921.19589041095
#print(y.std()) #79442.50288288662
#print(np.sqrt(mean_squared_error(y, y_pred))) #21683.845047497947
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(y, y_pred)) #13740.035023121582
#print(log_model.score(X, y)) #0.9254471384792963
log_model2 = LinearRegression().fit(X_train, y_train)
y_pred2 = log_model2.predict(X_train)






#print(mean_squared_error(y_train, y_pred2)) #443375207.01566803
#print(log_model.score(X_train, y_train)) #0.9257162775960132
#y_pred3 = log_model2.predict(X_test)
#print(mean_squared_error(y_test, y_pred3)) #2294930187.3854246
#print(log_model.score(X_test, y_test)) #0.924273864093837
#print(np.mean(np.sqrt(-cross_val_score(X, y, cv=5, scoring="neg_mean_squared_error"))))


########################################################################################################
# se the feature_importance function that indicates the importance level of the variables to plot the
# ranking of the features.
########################################################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


#model = LGBMRegressor().fit(X, y)
#plot_importance(model, X)


########################################################################################################
# Predict the empty salePrice variables in the test dataframe and create a dataframe suitable for
# submitting to the Kaggle page. (Id, SalePrice)
########################################################################################################

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(test_df.drop(["Id", "SalePrice"], axis=1))

dictionary = {"Id": test_df.index, "SalePrice": predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)