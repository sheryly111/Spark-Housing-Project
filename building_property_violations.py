import pandas as pd 

df = pd.read_csv("building_and_property_violations.csv")


#filter by years intersted
years_interest = ["2024", "2023", "2022", "2021", "2020", "2019", "2018"]

pattern = '|'.join(years_interest)
df_years = df[df['status_dttm'].str.contains(pattern, case=False, na=False)]

df_new = df.drop(['contact_addr1', 'contact_addr2', "contact_city", "contact_state", "contact_zip", "sam_id", "latitude", "longitude", "location"], axis=1) 

df_new['full_address'] = df_new['violation_stno'].str.cat(df_new['violation_street'], sep=' ').str.cat(df_new['violation_suffix'], sep=' ').str.cat(df_new['violation_city'], sep=' ').str.cat(df_new['violation_zip'], sep=' ')

df_new = df_new.drop(['violation_stno', 'violation_suffix', 'violation_street', 'violation_city', 'violation_state', 'violation_zip'], axis=1)

df_new.to_csv("cleaned_building_and_property_violations.csv", index=False)