# phys-gmm-python

Notes for first Edition: Only implement for datasets of 2D and option 'full'.
                         Most code follow the style like Matlab, use class to mimic the struct
                         Z_C means which table the member sits, Z_C[0] = 1 means the first data point sits at table 1
                         C means which datapoint current datapoint sit with, C[0] = 1 means first data point (access by index 0) sits with itself
                         if you want to access table member, likelihood, the index should be minus 1, table_Logliks[0] stores the likelihood of table 1
