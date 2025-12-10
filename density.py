# %% [markdown]
# # Student Densities by Neighborhood

# %%
#import libraries 
import pandas as pd
import matplotlib.pyplot as plt 

# %%
#open addresses csv 
df = pd.read_csv("addresses.csv")

df.head()

# %%
#extract zipcodes
df_zip = df[["6e. zip"]]

def zip_standard(zip): 
    if pd.isna(zip):
        return ""

    zip = str(zip).strip()

    if len(zip) >= 5:
        zip = zip[:5]
        if zip[-1] == ".":
            zip = zip[:-1]
    if len(zip) == 4:
        zip = "0" + zip 
    return zip

#apply zip_standard to the correct zipcode format: 
df_zip["6e. zip"] = df_zip["6e. zip"].apply(zip_standard)
#rename column to something better!!!!: 
df_zip = df_zip.rename(columns={"6e. zip": "Zipcode"})
df_zip.head()



# %%
'''
NOTES :
    counts the num occurances for each zip code, and then matches it to the zipcode 
    value_counts only works on a series, we want to "reset" the series so it matches the df 
'''
df_pop_zip = df_zip["Zipcode"].value_counts().reset_index() 

df_pop_zip.head()

# %%
df_zip_names = pd.read_csv("zippies.csv")

#standardize zipcodes using zip_standard
df_zip_names["Zipcode"] = df_zip_names["Zipcode"].apply(zip_standard)

df_zip_names.head()


# %%
df_zip_name_pop = pd.merge(df_pop_zip, df_zip_names, on="Zipcode", how="inner")
df_zip_name_pop.head()

#combine dorchesters bc multiple zipcodes for Dorchester
new_row = pd.DataFrame({"Neighborhood" : "Dorchester", "Zipcode": "0225/02124/02122/02121", "count" : [20108] })
df_zip_name_pop = pd.concat([df_zip_name_pop, new_row], ignore_index=True)
df_zip_name_pop = df_zip_name_pop.drop([14,7, 16, 19]) #drop old dorchesters 
print(df_zip_name_pop)

# %%
#get neighborhood sizes from other csv!!!
df_shapes = pd.read_csv("Boston_Neighborhood_Boundaries.csv")

#choose sqmiles and name columns
df_size = df_shapes[["sqmiles", "name"]] 

#combine neighborhoods that share zipcodes 
new_row1 = pd.DataFrame({"sqmiles": [0.66], "name" : "Back Bay/Bay Village/Theater District"})
new_row2 = pd.DataFrame({"sqmiles": [0.14], "name" : "Chinatown/Leather District"})

df_size = pd.concat([df_size, new_row1], ignore_index=True)
df_size = pd.concat([df_size, new_row2], ignore_index=True)

df_size.drop([4,5,6,14]) #drop old 

df_size = df_size.rename(columns={"name":"Neighborhood"})

print(df_size)

# %%
df_merged = pd.merge(df_size, df_zip_name_pop, on="Neighborhood", how="inner")
print(df_merged)

# %%
#calculate densities
density = []
for _, row in df_merged.iterrows():
    density.append(float(row[3])/row[0])

density_series = pd.Series(density)
df_merged["density"] = density_series
df_merged.head()

# %%
#PLOT PLOT PLOT !!!!
df_merged.plot(x= "Neighborhood", y="density", kind = "barh")
plt.title("Student Density per Boston Neighborhood")
plt.ylabel("Neighborhood")
plt.xlabel("Students per Sq. Miles")
plt.savefig("density_bar_graph")
plt.show()

# %%
#PLOT PLOT PLOT !!!!
colors = ["#C35454","#db913e", "#dfd67a", "#66bb60", "#50a580", "#78d8d0", "#4d6c93", "#887bdc", "#c68ddb", "#d375b7", "#974144", "#6b3f3f", "#725025", "#747E27", "#267d20", "#2a6486", "#2c225a", "#5d204c", "#E67A7A", ]
plot = df_merged.plot.pie(y='density', labels=None, figsize=(8, 8),colors = colors, startangle=90)
plt.legend(
    labels=df_merged['Neighborhood'],
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)
plt.title('Density of Students in Boston Neighborhoods')
#plt.ylabel('') # Remove the default 'Value' label on the y-axis
plt.savefig("density_pie_chart")
plt.show()


