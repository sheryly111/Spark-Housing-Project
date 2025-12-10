import pandas as pd

years = ["2024", "2023", "2022", "2021", "2020", "2019", "2018"]
df_list = []

for year in years:
    df = pd.read_csv(f"./property_assessment/{year}.csv")
    df["Year"] = year
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv("raw_property_assessment.csv", index=False)

df = pd.read_csv("raw_property_assessment.csv")

cols_keep = ["PID", "CM_ID", "GIS_ID", "ST_NUM", "ST_NAME", "UNIT_NUM", "CITY", "ZIP_CODE", "OWNER", "INT_COND", "EXT_COND", "OVERALL_COND", "BDRM_COND", "HEAT_TYPE", "AC_TYPE", "BED_RMS"]

df_new = df[cols_keep] 

def to_str_int_first(x):
    if pd.isna(x):
        return ""
    # convert float to int if it's an integer value (123.0 → 123)
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    # convert actual ints directly
    if isinstance(x, int):
        return str(x)
    # everything else to string, trimmed
    return str(x).strip()

cols = ["ST_NUM", "ST_NAME", "CITY", "ZIP_CODE"]

for c in cols:
    df_new[c] = df_new[c].apply(to_str_int_first)

df_new["full_address"] = (
    df_new["ST_NUM"] + " " + 
    df_new["ST_NAME"] + " " +
    df_new["CITY"] + " " + 
    df_new["ZIP_CODE"]
).str.replace(r"\s+", " ", regex=True).str.strip()

df_cleaned = df_new.drop(cols, axis=1)

df_cleaned.to_csv("cleaned_property_assessment.csv", index=False)