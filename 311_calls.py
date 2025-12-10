import pandas as pd 
import glob 

path = './311_calls/'  
all_files = glob.glob(path + "/*.csv")

combined_df = pd.concat(map(pd.read_csv, all_files), ignore_index=True)

combined_df.to_csv("raw_311_calls.csv", index=False)

df = pd.read_csv("raw_311_calls.csv")

columns_to_remove = ["submitted_photo", "closed_photo", "latitude", "longitude", "geom_4326", "source", "fire_district", "city_council_district", "police_district", "precinct", "location_street_name", "location_zipcode"] 

df = df.drop(columns=columns_to_remove)


df.to_csv("cleaned_311_calls.csv", index=False)


