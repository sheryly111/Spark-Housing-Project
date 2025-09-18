# Spark-Housing-Project

# Description

This project is Spark’s University Accountability Ordinance Project. By better understanding where students live, and what the housing conditions are like, we hope to understand how housing affordability is impacted and use that information to inform land use decisions. 

# Goals
* Better understand the geographical distribution of off-campus students throughout the city of Boston, such as districts where there might exist a large density of students as well as areas where student renters may be sparse, in addition to their respective housing conditions.
* Predict and identify problematic landlords from the demographic of renters, living conditions, and number previous housing violations.
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
We currently plan to use clustering. 


# What is your test plan? 
Withhold a 20% of the data for testing. 
