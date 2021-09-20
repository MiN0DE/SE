# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:56:14 2021

@author: MiN0DE
"""
import numpy as np
import matplotlib.pyplot as plt
year = [1999, 2020, 2030,2400]
pop = [3, 5, 2.5 , 10]
# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Make a line plot: year on the x-axis, pop on the y-axis
x = plt.plot(year,pop)

# Display the plot with plt.show()
plt.show()
#########################################################

# Change the line plot below to a scatter plot
plt.scatter(pop, year)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()
#########################################################

############## Histogram #################################
# Build histogram with 5 bins
plt.hist(pop,5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(pop,20)

# Show and clean up again
plt.show()
plt.clf()

##########################################################
########### Customization ################################

# Basic scatter plot, log scale
plt.scatter(pop, year)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Definition of tick_val and tick_lab
tick_val = [1, 10, 100]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()

# Specify c and alpha inside plt.scatter()
plt.scatter(x = pop, y =year, s = np.array(pop) * 2, alpha = 0.8)
plt.grid(True)
plt.show()

#########################################################
########### Dicionaries #################################

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index('germany')

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = { 'spain':'madrid', 'france' : 'paris','germany' : 'berlin', 'norway': 'oslo'}

# Print europe
print(europe)
############################################################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])

###########################################################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)
###############################################################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del(europe['australia'])

# Print europe
print(europe)
###############################################################
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital':'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)
##############################################################
###################### Pandas ################################
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country': names,'drives_right': dr,'cars_per_cap' : cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)




