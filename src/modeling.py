import pickle
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.stats import expon

import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (plot_confusion_matrix,
                             classification_report,
                             roc_curve,
                             f1_score,
                             balanced_accuracy_score,
                             brier_score_loss,
                             precision_recall_fscore_support,
                             zero_one_loss)

def get_data(filename, pickle=False):
    '''
    input desired table name as a string
    returns table as pandas data frame
    '''
    if pickle:
        df = pd.read_pickle(f'../data/{filename}')
        return df

    df = pd.read_csv(f'../data/ga_archive/{filename}', sep = '|')
    return df

def clean_data(df):
    vmask = df['date_last_voted'] >= '2020-11-03'
    df['voted'] = vmask.map({True: int(1), False: int(0)})
    
    df = df.drop(['land_district','land_lot','status_reason','city_precinct_id','county_districta_name',
                'county_districta_value','county_districtb_name','county_districtb_value','city_dista_name',
                'city_dista_value','city_distb_name','city_distb_value','city_distc_name','city_distc_value',
                'city_distd_name','city_distd_value','party_last_voted','city_school_district_name','municipal_name',
                'municipal_code','ward_city_council_code','race_desc','residence_city','residence_zipcode',
                'county_precinct_id','city_school_district_value','senate_district','house_district',
                'judicial_district','commission_district','school_district','date_added','date_changed',
                'district_combo','last_contact_date','ward_city_council_name','date_last_voted','registration_date',
                'registration_number','voter_status'], axis=1)
    
    counties = ['Appling','Atkinson','Bacon','Baker','Baldwin','Banks','Barrow','Bartow','Ben_Hill','Berrien','Bibb',
            'Bleckley','Brantley','Brooks','Bryan','Bulloch','Burke','Butts','Calhoun','Camden','Candler','Carroll',
            'Catoosa','Charlton','Chatham','Chattahoochee','Chattooga','Cherokee','Clarke','Clay','Clayton','Clinch',
            'Cobb','Coffee','Colquitt','Columbia','Cook','Coweta','Crawford','Crisp','Dade','Dawson','De_Kalb',
            'Decatur','Dodge','Dooly','Dougherty','Douglas','Early','Echols','Effingham','Elbert','Emanuel','Evans',
            'Fannin','Fayette','Floyd','Forsyth','Franklin','Fulton','Gilmer','Glascock','Glynn','Gordon','Grady',
            'Greene','Gwinnett','Habersham','Hall','Hancock','Haralson','Harris','Hart','Heard','Henry','Houston',
            'Irwin','Jackson','Jasper','Jeff_Davis','Jefferson','Jenkins','Johnson','Jones','Lamar','Lanier',
            'Laurens','Lee','Liberty','Lincoln','Long','Lowndes','Lumpkin','Macon','Madison','Marion','McDuffie',
            'McIntosh','Meriwether','Miller','Mitchell','Monroe','Montgomery','Morgan','Murray','Muscogee','Newton',
            'Oconee','Oglethorpe','Paulding','Peach','Pickens','Pierce','Pike','Polk','Pulaski','Putnam','Quitman',
            'Rabun','Randolph','Richmond','Rockdale','Schley','Screven','Seminole','Spalding','Stephens','Stewart',
            'Sumter','Talbot','Taliaferro','Tattnall','Taylor','Telfair','Terrell','Thomas','Tift','Toombs','Towns',
            'Treutlen','Troup','Turner','Twiggs','Union','Upson','Walker','Walton','Ware','Warren','Washington',
            'Wayne','Webster','Wheeler','White','Whitfield','Wilcox','Wilkes','Wilkinson','Worth']
    
    keys = range(1,161)
    county_dict = {}
    for key in keys:
        for county in counties:
            county_dict[key] = county
            counties.remove(county)
            break
            
    df['county_code'] = df['county_code'].replace(county_dict)
    df = df.rename(columns={'county_code': 'county'})
    
    rural = ['Appling', 'Atkinson','Bacon','Baker','Baldwin','Banks','Ben_Hill','Berrien','Bleckley','Brantley','Brooks',
         'Bryan','Burke','Butts','Calhoun','Candler','Charlton','Chattahoochee','Chattooga','Clay','Clinch','Coffee',
         'Colquitt','Cook','Crawford','Crisp','Dade','Dawson','Decatur','Dodge','Dooly','Early','Echols','Elbert',
         'Emanuel','Evans','Fannin','Franklin','Gilmer','Glascock','Grady','Greene','Habersham','Hancock','Haralson',
         'Harris','Hart','Heard','Irwin','Jasper','Jeff_Davis','Jefferson','Jenkins','Johnson','Jones','Lamar',
         'Lanier', 'Laurens','Lee','Lincoln','Long','Lumpkin','Macon','Madison','Marion','McDuffie','McIntosh',
         'Meriwether','Miller','Mitchell','Monroe','Montgomery','Morgan','Murray','Oconee','Oglethorpe','Peach',
         'Pickens','Pierce','Pike','Polk','Pulaski','Putnam','Quitman','Rabun','Randolph','Schley','Screven',
         'Seminole','Stephens','Stewart','Sumter','Talbot','Taliaferro','Tattnall','Taylor','Telfair','Terrell',
         'Thomas','Tift','Toombs','Towns','Treutlen','Turner','Twiggs','Union','Upson','Ware','Warren','Washington',
         'Wayne','Webster','Wheeler','White','Wilcox','Wilkes','Wilkinson','Worth']
    
    urban = ['Barrow','Bartow','Bibb','Bulloch','Carroll','Catoosa','Chatham','Cherokee','Clarke','Clayton','Cobb',
         'Columbia','Coweta','De_Kalb','Dougherty','Douglas','Effingham','Fayette','Floyd','Forsyth','Fulton','Glynn',
         'Gordon','Gwinnett','Hall','Henry','Houston','Jackson','Lowndes','Muscogee','Newton','Paulding','Richmond',
         'Rockdale','Spalding','Troup','Walker','Walton','Whitfield']
    
    military = ['Camden','Liberty']
    
    r_dummies = pd.get_dummies(df['race'], dtype='int64')
    df[r_dummies.columns] = r_dummies
    df = df.drop(['race'], axis=1)
    
    g_dummies = pd.get_dummies(df['gender'], dtype='int64')
    df[g_dummies.columns] = g_dummies
    df = df.drop(['gender'], axis=1)
    
    cd_dummies = pd.get_dummies(df['congressional_district'], prefix='cd', dtype='int64')
    df[cd_dummies.columns] = cd_dummies
    df = df.drop(['congressional_district'], axis=1)
    
    df['age'] = 2020 - df['birthyear']
    df['age'] = df['age'].astype('int64')
    df = df.drop(['birthyear'], axis=1)
    
    r_mask = df['county'].isin(rural)
    u_mask = df['county'].isin(urban)
    m_mask = df['county'].isin(military)

    df['rural'] = r_mask
    df['urban'] = u_mask
    df['military'] = m_mask

    df['rural'] = df['rural'].map({True: int(1), False: int(0)})
    df['urban'] = df['urban'].map({True: int(1), False: int(0)})
    df['military'] = df['military'].map({True: int(1), False: int(0)})

    df = df.drop('county', axis=1)
    
    return df

def get_X_y(clean_df, split = False):
    X = train
    y = X.pop('voted')

    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify = y)
        return X_train, X_test, y_train, y_test
    
    return X, y

def get_statsmodels_logit_model(X,y, grouping=5):
    groupings = ['y~ age',
                 'y~ C(AI) + C(AP) + C(BH) + C(HP) + C(OT) + C(U) + C(WH)',
                 'y~ C(F) + C(M) + C(O)',
                 'y~ C(rural) + C(urban) + C(military)',
                 'y~ C(cd_1) + C(cd_2) + C(cd_3) + C(cd_4) + C(cd_5) + C(cd_6) + C(cd_7) + C(cd_8) + C(cd_9) + C(cd_10) + C(cd_11) + C(cd_12) + C(cd_13) + C(cd_14) + C(cd_99999)',
                 'y~ C(AI) + C(AP) + C(BH) + C(HP) + C(OT) + C(U) + C(WH) + C(F) + C(M) + C(O) + C(rural) + C(urban) + C(military) + C(cd_1) + C(cd_2) + C(cd_3) + C(cd_4) + C(cd_5) + C(cd_6) + C(cd_7) + C(cd_8) + C(cd_9) + C(cd_10) + C(cd_11) + C(cd_12) + C(cd_13) + C(cd_14) + C(cd_99999) + age']
    
    model= smf.logit(formula=groupings[grouping], data= X).fit()
    return model.summary()

def get_best_models(X_train, y_train):
    logistic_grid = {'C':[0.01,0.03, 0.8, 2, 10, 50],
                 'solver':['liblinear'],
                 'max_iter' : [50],
                 'class_weight':['balanced', None],
                 'penalty':['l1', 'l2']}

    random_forest_grid = {'max_depth': [2, 4, 8],
                        'max_features': ['sqrt', 'log2', None],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 4],
                        'bootstrap': [True, False],
                        'class_weight': ['balanced'],
                        'n_estimators': [5,10,25,50,100,200]}

    grad_boost_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'max_depth': [2, 4, 8],
                    'subsample': [0.25, 0.5, 0.75, 1.0],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [5,10,25,50,100,200]}

    log_clf_rcv = RandomizedSearchCV(LogisticRegression(),
                                 logistic_grid,
                                 n_jobs=-1,
                                 verbose=False,
                                 refit = 'roc_auc',
                                 scoring=['roc_auc', 'f1', 'balanced_accuracy'])

    rf_clf_rcv = RandomizedSearchCV(RandomForestClassifier(),
                                random_forest_grid,
                                n_jobs=-1,
                                verbose=False,
                                    refit = 'roc_auc',
                                scoring=['roc_auc', 'f1', 'balanced_accuracy'],
                                random_state=0)

    grad_boost_rcv = RandomizedSearchCV(GradientBoostingClassifier(),
                                        grad_boost_grid,
                                        n_jobs=-1,
                                        verbose=False,
                                        refit = 'roc_auc',
                                        scoring=['roc_auc', 'f1', 'balanced_accuracy'])

    log_search = log_clf_rcv.fit(X_train, y_train)
    rf_search = rf_clf_rcv.fit(X_train, y_train)
    grad_boost_search = grad_boost_rcv.fit(X_train, y_train)

    best_log = log_search.best_estimator_
    best_rf = rf_search.best_estimator_
    best_gb = grad_boost_search.best_estimator_

    return best_log, best_rf, best_gb

if __name__=='__main__':
    pass