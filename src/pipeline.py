import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)

def get_data(filename, pickle=False):
    '''
    input desired table name as a string
    returns table as pandas data frame
    '''
    if pickle:
        df = pd.read_pickle(f'../data/{filename}')
        return df

    df = pd.read_csv(f'../data/ga_archive/{tablename}', sep = '|')
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

def clean_data_for_eda(df):
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
    
    df = df.replace(rural, 'Rural')
    df = df.replace(urban, 'Urban')
    df = df.replace(military, 'Military')
    
    df['age'] = 2020 - df['birthyear']
    df['age'] = df['age'].astype('int64')
    df = df.drop(['birthyear'], axis=1)
    
    df.congressional_district = df.congressional_district.astype('str')
    
    return df



if __name__=='__main__':
    pass