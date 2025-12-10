# %% [markdown]
# # Cleaning Student Addresses and SAM CSV, formatting

# %%
#import libraries
import pandas as pd
#open addresses csv
df = pd.read_csv("StudentAddresses-2016-2024.xlsx - Sheet1.csv")
df2 = pd.read_csv("live_street_address_management_sam_addresses.csv")
df.head()
print(df2.shape)
df2.head()

# %%
#extracting ears of interest, leases starting from 2018 to 2024
years_of_interest = [ "2019", "2020", "2021", "2022", "2023", "2024"]
pattern = '|'.join(years_of_interest)
matching_rows = df[df['year'].str.contains(pattern, case=False, na=False)]
SAM_matching_rows = df2[df2['last_edited_date'].str.contains(pattern, case=False, na=False)]    

# Combine all address cols with a space delimiter
matching_rows['full address'] = matching_rows['6a. street #'] + ' ' + matching_rows['6b. street name']+ ' ' + matching_rows['6c. street suffix']

SAM_matching_rows = SAM_matching_rows.drop(columns = ['shape_wkt', 'POINT_X', 'POINT_Y', 'X_COORD', 'Y_COORD'])
matching_rows = matching_rows.drop(columns=['6a. street #', '6b. street name', '6c. street suffix'])


display(matching_rows)
display(SAM_matching_rows)
print(SAM_matching_rows.shape)
unique_unis = df['university'].unique()
print(unique_unis)

#test_row = df.loc[(df['6a. street #'] == '10')& (df['6b. street name'] == 'Higgins') & (df['6c. street suffix'] == 'ST')	]
#display(test_row)

matching_rows.to_csv('cleaned_student_addresses.csv', index=False)

#one SAM ID FOR ONE LANDLORD
list(SAM_matching_rows)
#num unique landlords
unique_landlords = SAM_matching_rows['SAM_ADDRESS_ID'].nunique()
print(f'Number of unique landlords: {unique_landlords}')

# %%
#extracting ONLY sam ids from student addresses
matching_rows = matching_rows.rename(columns={'full address': 'FULL_ADDRESS'})
df_matched = matching_rows.merge(SAM_matching_rows, on='FULL_ADDRESS', how='inner')

#keep only cols we want
col_keep = ['6d. unit #',
 '6e. zip',
 '9. 5 or more undergrads/unit (y/n)',
 'year',
 'FULL_ADDRESS',
 'SAM_ADDRESS_ID',
 'BUILDING_ID',
 'STREET_NUMBER',
 'UNIT',
 'ZIP_CODE',
 'WARD',
 'PARCEL',
 'created_date',
 'last_edited_date']

df_matched = df_matched[col_keep]
#unique student housing addresse
num_unique_student_addresses = matching_rows['FULL_ADDRESS'].nunique()
print(f'Number of unique student housing addresses: {num_unique_student_addresses}')
list(df_matched.columns.values)
df_matched.head()

#to csv
df_matched.to_csv('cleaned_student_addresses_SAM.csv', index=False)



# %% [markdown]
# # Merging Everything
# 
# This section merges the following datasets into one coherent one we will use: 
# 1. cleaned_student_addresses_SAM.csv
# 2. cleaned_311_calls.csv
# 3. cleaned_building_and_property_violations.csv
# 4. cleaned_property_assessment.csv
# 

# %%
import pandas as pd
df_matched = pd.read_csv("cleaned_student_addresses_SAM.csv")
threeoneone = pd.read_csv("cleaned_311_calls.csv")
bpviolations = pd.read_csv("cleaned_building_and_property_violations.csv")
propassessments = pd.read_csv("cleaned_property_assessment.csv")

# %%
import re

suffix_map = {
    'street': 'st', 'st': 'st', 'st.': 'st',
    'road': 'rd',   'rd': 'rd', 'rd.': 'rd',
    'avenue': 'ave','ave': 'ave','ave.': 'ave',
    'boulevard': 'blvd', 'blvd': 'blvd',
    'drive': 'dr', 'dr': 'dr',
    'lane': 'ln', 'ln': 'ln',
    'court': 'ct','ct': 'ct',
    'place': 'pl','pl': 'pl',
    'circle': 'cir','cir': 'cir'
}


# Build a regex pattern to match street address ending with a known suffix
suffix_pattern = '|'.join(suffix_map.keys())
street_regex = re.compile(r'(\d+\s+[\w\s]+?\s+(?:' + suffix_pattern + r'))', flags=re.IGNORECASE)



def normalize_addr(addr):
    # lowercase & strip
    
    a = addr.lower().strip()
    
    # take everything BEFORE the first comma
    a = a.split(',')[0]
    
    # remove punctuation (except spaces and digits)
    a = re.sub(r'[^a-z0-9 ]', ' ', a)
    
    if pd.isna(addr):
        return ""
    match = street_regex.search(addr)
    if not match:
        return addr.lower().strip()  # fallback
    street = match.group(1).lower().strip()
    
    # standardize suffix
    parts = street.split()
    last = parts[-1]
    if last in suffix_map:
        parts[-1] = suffix_map[last]
    return " ".join(parts)


# %%
propassessments['full_address'] = propassessments['full_address'].apply(normalize_addr)
threeoneone['location'] = threeoneone['location'].apply(normalize_addr)
bpviolations['full_address'] = bpviolations['full_address'].astype(str).apply(normalize_addr)
df_matched['FULL_ADDRESS'] = df_matched['FULL_ADDRESS'].apply(normalize_addr)

#normalize names
propassessments = propassessments.rename(columns={'full_address': 'FULL_ADDRESS'})
bpviolations = bpviolations.rename(columns={'full_address': 'FULL_ADDRESS'})
threeoneone = threeoneone.rename(columns={'location': 'FULL_ADDRESS'})

bpviolations['status_dttm'] = bpviolations['status_dttm'].astype(str).str[:4]
bpviolations = bpviolations.rename(columns={'status_dttm':'year'})

threeoneone['open_dt'] = threeoneone['open_dt'].astype(str).str[:4]
threeoneone = threeoneone.rename(columns={'open_dt':'year'})

df_matched["year"] = df_matched["year"].astype(str).str[:4]

# type checking 
df_matched["year"] = df_matched["year"].astype(str).replace("nan", "")
threeoneone["year"] = threeoneone["year"].astype(str).replace("nan", "")
bpviolations["year"] = bpviolations["year"].astype(str).replace("nan", "")
propassessments["year"] = propassessments["year"].astype(str).replace("nan", "")

df_matched["FULL_ADDRESS"] = df_matched["FULL_ADDRESS"].astype(str).replace("nan", "")
threeoneone["FULL_ADDRESS"] = threeoneone["FULL_ADDRESS"].astype(str).replace("nan", "")
bpviolations["FULL_ADDRESS"] = bpviolations["FULL_ADDRESS"].astype(str).replace("nan", "")

# %%
print("threeoneone:")
display(threeoneone.head())

print("bpviolations:")
display(bpviolations.head())

print("propassessments:")
display(propassessments.head())

# %%
print(df_matched.shape)
print(propassessments.shape)
print(threeoneone.shape)

print(df_matched['FULL_ADDRESS'].duplicated().sum())
print(propassessments['FULL_ADDRESS'].duplicated().sum())
print(threeoneone['FULL_ADDRESS'].duplicated().sum())

# %%
df_merged_test = df_matched.copy() #copy just in case... 
df_address_311 = df_merged_test.merge(threeoneone, on=['FULL_ADDRESS', "year"], how='left')
df_add_propassessments = df_address_311.merge(propassessments, on=['FULL_ADDRESS', "year"], how='left')
df_merge_total = df_add_propassessments.merge(bpviolations, on=['FULL_ADDRESS', "year"], how='left')

print(df_merge_total.shape)
list(df_merge_total.columns.values)

# %%
final_drop = [
    '6d. unit #', 
    '6e. zip',
    'STREET_NUMBER', 
    'UNIT',
    'ZIP_CODE',
    '_id_x',
    'queue',
    'pwd_district',
    'neighborhood',
    'neighborhood_services_district',
    'ward_x',
    'UNIT_NUM',
    "violation_sthigh",
    'ward_y',
    '_id_y',
    'case_no',
    'ap_case_defn_key',
    'status',
    'code',
    'value',
    'description']
df_merged = df_merge_total.drop(final_drop, axis=1)

df_merged = df_merged.rename(columns={'9. 5 or more undergrads/unit (y/n)':'over_5'}) 
df_merged = df_merged.rename(columns={'FULL_ADDRESS':'full_address'}) 
df_merged = df_merged.rename(columns={'SAM_ADDRESS_ID':'sam_id'}) 
df_merged = df_merged.rename(columns={'BUILDING_ID':'building_id'}) 
df_merged = df_merged.rename(columns={'WARD':'ward_id'}) 
df_merged = df_merged.rename(columns={'PARCEL':'parcel_num'}) 
df_merged = df_merged.rename(columns={'created_date':'case_created_date'}) 
df_merged = df_merged.rename(columns={'last_edited_date':'last_case_update'}) 
df_merged = df_merged.rename(columns={'sla_target_dt':'targeted_deadline'}) 
df_merged = df_merged.rename(columns={'closed_dt':'close_date'}) 
df_merged = df_merged.rename(columns={'on_time':'case_met_deadline'}) 
df_merged = df_merged.rename(columns={'subject':'case_subject'}) 
df_merged = df_merged.rename(columns={'reason':'case_reason'}) 
df_merged = df_merged.rename(columns={'type':'case_type'}) 
df_merged = df_merged.rename(columns={'department':'case_department'}) 
df_merged = df_merged.rename(columns={'PID':'p_id'}) 
df_merged = df_merged.rename(columns={'CM_ID':'cm_id'}) 
df_merged = df_merged.rename(columns={'GIS_ID':'gis_id'}) 
df_merged = df_merged.rename(columns={'OWNER':'landlord_name'}) 
df_merged = df_merged.rename(columns={'INT_COND':'int_cond'}) 
df_merged = df_merged.rename(columns={'EXT_COND':'ext_cond'}) 
df_merged = df_merged.rename(columns={'OVERALL_COND':'overall_cond'}) 
df_merged = df_merged.rename(columns={'BDRM_COND':'bdrm_cond'}) 
df_merged = df_merged.rename(columns={'HEAT_TYPE':'heat_type'}) 
df_merged = df_merged.rename(columns={'AC_TYPE':'ac_type'}) 
df_merged = df_merged.rename(columns={'BED_RMS':'num_bed_rms'}) 

list(df_merged.columns.values)

print(df_merged.isna().sum())

# %%
df_merged.to_csv("raw_merged.csv")


