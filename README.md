# Spark-Housing-Project

# Description

This project is Spark’s University Accountability Ordinance Project. By better understanding where students live, and what the housing conditions are like, we hope to understand how housing affordability is impacted and use that information to develop effective regulations to minimize the possibility of exploitative landlords. 

# Goals
Evaluate data from the past five years in order to: 
* Better understand the geographical distribution of off-campus students throughout the city of Boston, ranking neighborhoods most dense to least. 
  * Identify how housing conditions differ between neighborhoods with different student population densities. Split the dataset into three: dense, average, sparse. 
  * Housing conditions might include: ratio of rental fees to number of bedrooms, number of violations, and how problematic the landlord is.
* Predict and identify exploitative landlords through comparing the frequency, type, and urgency of 311 service requests from neighborhoods throughout the city of Boston. Exploitative landlords shall be defined as those with more than 5 urgent service requests and more than 5 violations in the past 5 years. 
* Create a publicly accessible database that facilitates the process of ensuring compliance and promoting responsible property management.


# What data needs to be collected 
* Percentage of student renters in each district 
  * [Student Housing Data](https://docs.google.com/spreadsheets/d/11X4VvywkSodvvTk5kkQH7gtNPGovCgBq/edit?gid=1139465182#gid=1139465182)
  * [Neighborhood boundaries](https://data.boston.gov/dataset/boston-neighborhood-boundaries-approximated-by-2020-census-tracts) 
* Housing conditions of student renters 
  * How many students per unit 
  * Access to resources (heat, water, etc…) 
  * [Known housing violations](https://data.boston.gov/dataset/building-and-property-violations1/resource/800a2663-1d6a-46e7-9356-bedb70f5332c)
  * [311 Service Requests](https://data.boston.gov/dataset/311-service-requests) (frequency, type, and urgency of complaints)  

# How you plan on modeling the data
* Heatmap for Neighborhood & landlord quality 
* Scatterplot for Rent prices vs Number of Bedrooms 
* Density Graph for Neighborhood & Number of Students 
* Histogram for:
  *  Number of violations in each neighborhood
  *  Number of occurrences for each violation 

# What is your test plan? 
Split data from the past 5 years into training/test/validation (70/15/15).
