import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# What would the fan speed be in the following circumstance:
T = 28
H = 37

# Generate universe variables
#   * Temperature in the subjective range [-10, 50] degC
#   * Humidity in the subjective range [0, 100] percent
#   * Fan has a range of [0, 100] percent of speed
x_temp = np.arange(-10, 50, 1) 
x_humidity = np.arange(0, 100, 1) 
x_fan  = np.arange(0, 100, 1)

# Generate fuzzy membership functions
temp_cold = fuzz.gaussmf(x_temp, 10, 5.5)
temp_good = fuzz.gaussmf(x_temp, 21, 5)
temp_hot = fuzz.gaussmf(x_temp, 33, 6.5)
humid_dry = fuzz.trimf(x_humidity, [0, 0, 50])
humid_acceptable = fuzz.trimf(x_humidity, [20, 50, 80])
humid_wet = fuzz.trimf(x_humidity, [50, 100, 100])
fan_lo = fuzz.trimf(x_fan, [0, 0, 50])
fan_md = fuzz.trimf(x_fan, [0, 50, 100])
fan_hi = fuzz.trimf(x_fan, [50, 100, 100])

################################################################################
# Visualize these universes and membership functions

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 3.5))

ax0.plot(x_temp, temp_cold, 'b', linewidth=1.5, label='Cold')
ax0.plot(x_temp, temp_good, 'g', linewidth=1.5, label='Good')
ax0.plot(x_temp, temp_hot, 'r', linewidth=1.5, label='Hot')
ax0.plot([T, T], [0, 1], ':k')
ax0.set_xlabel('Temperature ($^o$C)')
ax0.legend(loc='upper left')

ax1.plot(x_humidity, humid_dry, 'b', linewidth=1.5, label='Dry')
ax1.plot(x_humidity, humid_acceptable, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(x_humidity, humid_wet, 'r', linewidth=1.5, label='Wet')
ax1.plot([H, H], [0, 1], ':k')
ax1.set_xlabel('Humidity (%)')
ax1.legend(loc='upper left')

ax2.plot(x_fan, fan_lo, 'b', linewidth=1.5, label='Low')
ax2.plot(x_fan, fan_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_fan, fan_hi, 'r', linewidth=1.5, label='High')
ax2.set_xlabel('Fan speed (%)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

################################################################################

# We need the activation of our fuzzy membership functions at these values.
# The exact values T and H do not exist on our universes...
# This is what fuzz.interp_membership exists for! find the fuzzy value in each mem. func.
temp_level_lo = fuzz.interp_membership(x_temp, temp_cold, T)
temp_level_md = fuzz.interp_membership(x_temp, temp_good, T)
temp_level_hi = fuzz.interp_membership(x_temp, temp_hot, T)

humid_level_lo = fuzz.interp_membership(x_humidity, humid_dry, H)
humid_level_md = fuzz.interp_membership(x_humidity, humid_acceptable, H)
humid_level_hi = fuzz.interp_membership(x_humidity, humid_wet, H)

##### Rule 1 - If the Temperature is Hot AND Humidity is Wet, then the fan is High
# The AND operator means we take the minimum of these two.
alpha1 = np.fmin(temp_level_hi, humid_level_hi)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
# This finds the top of the fuzzy patch for high fan
fan_activation_hi = np.fmin(alpha1, fan_hi)  # removed entirely to 0

##### Rule 2 - If the Humidity is Acceptable, then the fan is Medium
alpha2 = humid_level_md

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
# This finds the top of the fuzzy patch for medium fan
fan_activation_md = np.fmin(alpha2, fan_md)

##### Rule 3 - If the Temperature is Good OR the Humidity is Dry, then the fan will be Low
# The OR operator means we take the maximum of these two.
alpha2 = np.fmax(temp_level_md, humid_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
# This finds the top of the fuzzy patch for low fan
fan_activation_lo = np.fmin(alpha2, fan_lo)

# Visualize this
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 3.5))
fan0 = np.zeros_like(x_fan)

ax0.plot(x_temp, temp_cold, 'b', linewidth=1.5, label='Cold')
ax0.plot(x_temp, temp_good, 'g', linewidth=1.5, label='Good')
ax0.plot(x_temp, temp_hot, 'r', linewidth=1.5, label='Hot')
ax0.plot([T, T], [0, 1], ':k')
ax0.plot([T, 50], [temp_level_hi, temp_level_hi], '-m', label = 'rule 1')
ax0.plot([T, 50], [temp_level_md, temp_level_md], '--y', label = 'rule 3')
ax0.set_xlabel('Temperature ($^o$C)')
ax0.legend(loc='upper left')

ax1.plot(x_humidity, humid_dry, 'b', linewidth=1.5, label='Dry')
ax1.plot(x_humidity, humid_acceptable, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(x_humidity, humid_wet, 'r', linewidth=1.5, label='Wet')
ax1.plot([H, H], [0, 1], ':k')
ax1.plot([H, 100], [humid_level_hi, humid_level_hi], '-m', label = 'rule 1')
ax1.plot([H, 100], [humid_level_md, humid_level_md], '-.c', label = 'rule 2')
ax1.plot([H, 100], [humid_level_lo, humid_level_lo], '--y', label = 'rule 3')
ax1.set_xlabel('Humidity (%)')
ax1.legend(loc='upper left')

ax2.plot(x_fan, fan_lo, 'b', linewidth=0.5, linestyle='--', label='Low')
ax2.plot(x_fan, fan_md, 'g', linewidth=0.5, linestyle='--', label='Medium')#, marker = '.')
ax2.fill_between(x_fan, fan0, fan_activation_hi, facecolor='r', alpha=0.7, label='rule 1')
ax2.fill_between(x_fan, fan0, fan_activation_md, facecolor='g', alpha=0.7, label='rule 2')
ax2.fill_between(x_fan, fan0, fan_activation_lo, facecolor='b', alpha=0.7, label='rule 3')
ax2.plot(x_fan, fan_hi, 'r', linewidth=0.5, linestyle='--', label='High')
ax2.set_xlabel('Fan speed (%)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()


################################################################################

# Aggregate all three output membership functions together
beta = np.fmax(fan_activation_lo,
                     np.fmax(fan_activation_md, fan_activation_hi))

# Calculate defuzzified result
fan_cog = fuzz.defuzz(x_fan, beta, 'centroid')
fan_mom = fuzz.defuzz(x_fan, beta, 'mom')
fan_som = fuzz.defuzz(x_fan, beta, 'som')
fan_lom = fuzz.defuzz(x_fan, beta, 'lom')
fan_cog_activation = fuzz.interp_membership(x_fan, beta, fan_cog)  # for plot
fan_mom_activation = fuzz.interp_membership(x_fan, beta, fan_mom)  # for plot
fan_som_activation = fuzz.interp_membership(x_fan, beta, fan_som)  # for plot
fan_lom_activation = fuzz.interp_membership(x_fan, beta, fan_lom)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_fan, fan_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_fan, fan_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_fan, fan_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_fan, fan0, beta, facecolor='Orange', alpha=0.7)
ax0.plot([fan_cog, fan_cog], [0, fan_cog_activation], 'k', linewidth=1.5, alpha=0.9, label = 'Centorid')
ax0.plot([fan_mom, fan_mom], [0, fan_mom_activation], 'm', linewidth=1.5, alpha=0.9, label = 'MOM')
ax0.plot([fan_som, fan_som], [0, fan_som_activation], 'y', linewidth=1.5, alpha=0.9, label = 'SOM')
ax0.plot([fan_lom, fan_lom], [0, fan_lom_activation], 'c', linewidth=1.5, alpha=0.9, label = 'LOM')
ax0.legend()

plt.show()


