# Spark-Housing-Project Midterm Report

# Data Preprocessing Methods Applied 
At this point in our project, we:
* Pulled zipcodes of student addresses from Student Housing Data 
* Standardized the zipcodes by 
  * concatenating “0” in front of zipcodes with 4 digits 
  * truncated zipcodes with over 5 digits
* Counted instances of all unique zipcodes and stored them into a new DataFrame (df_pop_zip)
* Made a csv file containing neighborhoods of interest (greater Boston area) and mapped their corresponding zipcodes (zippies.csv)
* Standardized the new zippies.csv using the same method defined above and named the new DataFrame df_zip_names
* Combined df_pop_zip with df_zip_names using the merge function to create a DataFrame containing information about 
  * Zipcode
  * Population
  * Neighborhood
* Pulled the sizes (by square miles) of neighborhoods of interest from Boston Neighborhood Boundaries into df_sizes
* Iterated through the rows of the merged DataFrame and df_sizes to calculate density of students per square mile
* Added density as a column into the merged DataFrame

# Data Modeling Methods Applied & Preliminary Results 
We have chosen to use a horizontal bar graph to represent the densities of college students living in each neighborhood in the greater Boston area by a metric of students per square miles. 

To visually get a better sense of the distribution compared to the entire population of students living in the greater Boston area, we also modeled the same data using a pie chart.

By looking at the data visualizations, we can see that the neighborhoods with the greatest student population density are Mission Hill and Chinatown/Leather District, and the least densely populated areas being West Roxbury and Hyde Park.

In the future, we are planning on:
* Splitting the neighborhoods into three groups: sparse, average, dense student populations and comparing these groups to datasets containing housing violations and 311 service requests. This way, we will be able to observe exploitative trends that occur between neighborhoods with different student population densities.
* Using K-means clustering to find patterns between density, rent, and violations. This would reveal the most important features to consider when creating a model that will be able to accurately predict exploitative landlords.
