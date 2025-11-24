# README
## 1 Overview
This is a project to model a new E-bike hire scheme for Edinburgh. Through the analysis of historical data and weighted decision variables, 
the model aims to plan the deployment of stations across 250 generated clusters across three time periods. 
### 1.1 Built with:
- Python 3.12.11 with Python Mosel API

### 1.2 The Team
Ben de Sousa s2892249@ed.ac.uk<br>
Alexandros Zachakos s2899095@ed.ac.uk<br>
Nolan Willoughby s2822750@ed.ac.uk<br>
Todd House s2809867@ed.ac.uk

## 2 Packages Required
- pandas 2.3.3
- xpress 9.7.0
- numpy 2.3.4
- matplotlib 3.10.7
- sklearn 1.7.2

## 3 Auxiliary Scripts
These are supporting scripts required for the running of the models. We recommend that you run all of them besides 2.1, to ensure full model functionality.
### 3.1 mmcsProjectPOIKmeans.py
Runs the clustering alorgithm to group together Points of Interest (POIs) and generate clusters from these groupings.
### 3.2 station_assingments.py
Assigns historic stations (and thus their trip data) to the nearest cluster geographically
### 3.3 m_matrix.py
Builds the matrix of trip data for each of the clusters. All clusters with no historic stations are assigned the average trips.
### 3.4 building_counts.py
Counts the number of each type of POI (building) in each cluster

## 4 The Models
### 4.1 basic.py
This is the basic, single time period model. This does not generate the final result 
but is the basis for the other models so is retained for clarity of the development process.
### 4.2 periodic_rollout.py
This is the final complete model. Calculates the rollout of the scheme over three time periods. Uses a assigns demand deterministically.
### 4.3 extension_model.py
Largely the same as model 3.2 however, it introduces a stochastic element to more accurately predict demand over the time periods.


