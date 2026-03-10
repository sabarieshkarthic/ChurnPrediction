import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
#model building requirements
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from imblearn.pipeline import Pipeline


def check_info(cdf):
    print(cdf.info())
    cdf['TotalCharges']=pd.to_numeric(cdf['TotalCharges'],errors='coerce')
    return cdf

def splitXY(cdf):
    X=cdf.drop('Churn',axis=1)
    Y=cdf['Churn']
    return X,Y

def split(X,Y):
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
    return X_train,X_test,y_train,y_test


def checkNull(X_train):
    print(X_train.isnull().sum())

def fillNull_and_TotalChargeDist(X_train,X_test):
    print(X_train['TotalCharges'].describe())
    #Boxplot for outlier verification
    plt.boxplot(X_train['TotalCharges'].dropna())
    plt.title("Box plot for Total Charges to understand distribution")
    plt.show()
    #comapring mean and median|mean>median so it is right skewed as well as max is double the times
    #larger than the 75th percentile,so it contains outliers||replacing Nan with median
    X_train['TotalCharges']=X_train['TotalCharges'].fillna(X_train['TotalCharges'].median())
    print(X_train.isnull().sum())
    #Xtest fill ##########
    X_test['TotalCharges']=X_test['TotalCharges'].fillna(X_train['TotalCharges'].median())
    return X_train,X_test


def churnDistribution(y_train):
    ratio=y_train.value_counts()
    percentages=[(ratio['No']/len(y_train))*100,(ratio['Yes']/len(y_train))*100]
    plt.bar(ratio.keys(), percentages, color=['blue', 'red'])
    plt.title("Churn Distribution %")
    plt.ylabel("Percentage")
    plt.show()
    
def BinningOnTenure(X_train,X_test):
    #Making bins for column:Tenure
    # QQ plot for identifying whether the tenure column is normal distributed
    #since it is not normally distributed we used quantile binning/equal frrequency instead of equal width because equal width
    #cannot handle skewed data 
    print("Max:",max(X_train['tenure']))
    print("Min:",min(X_train['tenure']))

    stats.probplot(X_train['tenure'].sort_values(), dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()

    #why 6 category coz 72 months converted into 6 years 72/12=6;
    X_train['tenure_Bins'], bin_edges = pd.qcut(X_train['tenure'],q=6,labels=[1,2,3,4,5,6],retbins=True,duplicates='drop')  # just in case some quantiles are identical)
    X_train.drop('tenure', axis=1, inplace=True)

    # binning in Xtest##########
    X_test['tenure_Bins'] = pd.cut(X_test['tenure'],bins=bin_edges,labels=[1,2,3,4,5,6],include_lowest=True)
    X_test=X_test.drop('tenure',axis=1)
    return X_train,X_test

def dropChurnID(X_train,X_test):
    X_train=X_train.drop('customerID',axis=1)
    print(X_train.columns)
    
    #dropping ID in Xtest##########
    X_test=X_test.drop('customerID',axis=1)
    print(X_test.columns)
    return X_train,X_test

#ED
#for iplot,feature in enumerate(cdf.drop(columns=['MonthlyCharges','TotalCharges','Churn'])):
#   plt.figure(iplot)
#    sns.countplot(data=cdf,x=feature,hue='Churn')

def mapping_And_OneHot(X_train,X_test,y_train,y_test):
    #mapping Churn 1 and 0#######

    y_train=y_train.map({'Yes':1,'No':0})
    print(y_train)

    #churn 1 0 mapping in Xtest###########
    y_test=y_test.map({'Yes':1,'No':0})
    print(y_test)
    #One-pot encoding
    
    X_train=pd.get_dummies(X_train)
    print(X_train.columns)
    
    #one hot enodong in Xtest########
    X_test=pd.get_dummies(X_test)
    print(X_test.columns)
    return X_train,X_test,y_train,y_test




def correlation(X_train,y_train):
    # corrlation for churn with all the feature
    # Combine features and target
    train_df=X_train.copy()
    train_df['Churn']=y_train

    # Compute correlations
    plt.figure(figsize=(20, 8))
    train_df.corr(numeric_only=True)['Churn'].sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Correlation with Churn')
    plt.ylabel('Correlation')
    plt.xlabel('Features')
    plt.show()
    #dummy variable trap
    #No need remove columns to prevent multicollinearity 
    #because decision tree doest assume independence and doesnt 
    #have Xtranspose *X to fail
    
def MonthyVsTotalCharges(X_train):
    plt.scatter(X_train['MonthlyCharges'],X_train['TotalCharges'])
    plt.xlabel("Monthly Charges")
    plt.ylabel("Total Charges")
    plt.show()

def MonthyChargesChurnVsNoChurn(X_train,y_train):
    #Monthly Charges churn vs No churn
    train_df=X_train.copy()
    train_df['Churn']=y_train
    A= sns.kdeplot(train_df.MonthlyCharges[train_df["Churn"] == 1],color="Red", shade=True)
    A= sns.kdeplot(train_df.MonthlyCharges[train_df["Churn"] == 0],color="Blue", shade=True)
    A.legend(["Churn","No Churn"],loc='upper right')
    A.set_ylabel('pdf')
    A.set_xlabel('Monthly Charges')
    A.set_title('Monthly charges by churn and No Churn')
    plt.show()
    #Note:As monthly charges increases -> churn rate increases

def TotalChargesChurnVsNoChurn(X_train,y_train):
    #Total Charges churn vs No churn
    train_df=X_train.copy()
    train_df['Churn']=y_train
    B= sns.kdeplot(train_df.TotalCharges[train_df["Churn"] == 1],color="Red", shade=True)
    B= sns.kdeplot(train_df.TotalCharges[train_df["Churn"] == 0],color="Blue", shade=True)
    B.legend(["Churn","No Churn"],loc='upper right')
    B.set_ylabel('pdf')
    B.set_xlabel('Total Charges')
    B.set_title('Total charges by churn and No Churn')
    plt.show()
    #Note: churn rate increases at lower total charges

    #from tenure,total,monthly charges we conclude that 
    #tenure bin 1 the churn rate is high at bin 6 churn rate is low
    #As monthly charges increases -> churn rate increases and 
    # lower monthly charges increase in no churn rate
    #churn rate increases at lower total charges and gradullay falls when total charges increases
    #No churn rate increases at lower total charges nd gradullay falls when total charges increases
    # hence ,Higher Monthly Charge, Lower Tenure and Lower Total Charge are linked to High Churn."

def heatmap(X_train,y_train):
    train_df=X_train.copy()
    train_df['Churn']=y_train
    plt.figure(figsize=(12,12))
    sns.heatmap(train_df.corr(),cmap="Paired");
    plt.show()
    
def Monthly_And_TotalOnChurn(X_train,y_train):
    train_df=X_train.copy()
    train_df['Churn']=y_train
    df = train_df[['MonthlyCharges', 'TotalCharges', 'Churn']].copy()
    sns.pairplot(df, hue='Churn')
    plt.show()
    
def ModelBuilding_And_Accuracy(X_train,X_test,y_train,y_test):
    baselineRF=RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        random_state=100,
        class_weight='balanced',
        n_jobs=-1
    )
    baselineRF.fit(X_train, y_train)
    y_pred=baselineRF.predict(X_test)

    print("Baseline - Random Forest")
    print(classification_report(y_test,y_pred,labels=[0, 1]))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


    smote=SMOTEENN(random_state=42)
    X_new,y_new=smote.fit_resample(X_train,y_train)
    rf_smote = RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        random_state=100,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_smote.fit(X_new, y_new)
    new_y_pred=rf_smote.predict(X_test)

    print("After SMOTEENN Resampling (Random Forest)")
    print(classification_report(y_test,new_y_pred,labels=[0, 1]))
    cm=confusion_matrix(y_test,new_y_pred)
    print(cm)

    train_df_new=X_new.copy()
    train_df_new['Churn']=y_new
    df=train_df_new[['MonthlyCharges','TotalCharges','Churn']].copy()
    sns.pairplot(df,hue='Churn')
    plt.title("After oversampling")
    plt.show()

    ratio_new=y_new.value_counts()
    percentages = [(ratio_new[0] / len(y_new)) * 100, (ratio_new[1] / len(y_new)) * 100]
    plt.bar(ratio_new.keys(), percentages, color=['blue', 'red'])
    plt.title("Churn Distribution %")
    plt.ylabel("Percentage")
    plt.show()

    pipeline_cv=Pipeline([
        ('smote',SMOTEENN(random_state=42)),
        ('clf',RandomForestClassifier(
            n_estimators=200,
            random_state=100,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    scores=cross_val_score(pipeline_cv,X_train,y_train,cv=5,scoring='f1_macro')
    print("Cross-Validated F1 Score (Train set only):",scores.mean())


    param_grid = {
        'clf__n_estimators': [100,200,300],
        'clf__max_depth': [5,10,15,None],
        'clf__min_samples_split': [2,5,10],
        'clf__min_samples_leaf': [1,2,4]
    }

    grid=GridSearchCV(pipeline_cv,param_grid,cv=5,scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train,y_train)`
    y_pred_best=grid.predict(X_test)
    print("After Hyperparameter Tuning (Random Forest)")
    print("Best Params:", grid.best_params_)
    print(classification_report(y_test,y_pred_best,labels=[0, 1]))
    cm=confusion_matrix(y_test,y_pred_best)
    print(cm)


    import pickle
    with open('model.pkl','wb') as f: 
        pickle.dump(grid.best_estimator_, f) 
    with open('columns.pkl','wb') as f: 
        pickle.dump(X_train.columns,f)
    with open('bin_edges.pkl','wb') as f: 
        pickle.dump(bin_edges,f)


def main():
    cdf=pd.read_csv("D:\SEM-5\churn pred\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cdf=check_info(cdf)
    
    X,Y=splitXY(cdf)
    X_train,X_test,y_train,y_test=split(X,Y)
    checkNull(X_train)
    X_train,X_test=fillNull_and_TotalChargeDist(X_train,X_test)
    churnDistribution(y_train)
    X_train,X_test=BinningOnTenure(X_train,X_test)
    X_train,X_test=dropChurnID(X_train,X_test)
    X_train,X_test,y_train,y_test=mapping_And_OneHot(X_train,X_test,y_train,y_test)
    correlation(X_train,y_train)
    MonthyVsTotalCharges(X_train)
    MonthyChargesChurnVsNoChurn(X_train,y_train)
    TotalChargesChurnVsNoChurn(X_train,y_train)
    heatmap(X_train,y_train)
    Monthly_And_TotalOnChurn(X_train,y_train)
    ModelBuilding_And_Accuracy(X_train,X_test,y_train,y_test)
    
    
    
        
    
    
    
    
        
    
    
if "__name__"=="__main__":
    main()


