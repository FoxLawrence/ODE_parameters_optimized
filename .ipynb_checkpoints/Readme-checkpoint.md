# Structe
y1 means WT
y2 means dPBS
y3 means dOCP

### code folder
**code** folder inculde all python code

### plot folder
**plot** folder have 3 folders, WT, dPBS, dOCP. All plots will be stored in the corresponding folders

### exp_data folder

**exp_data** folder include all exp_data and the file type should be .csv.
**All file should be named as \*_data.csv.**
The CSV file should have only two columns, and they should be named "time" and "values" respectively to ensure that the program can read it correctly.
### opt_par folder
**opt_par** folder like **plot** folder that stored all optimized parameters use .csv file.

## Code

### Optimize
optimize_*.py means this python program is specially used for * samples.

All .py have the same code and only a few differences.

For different samples the **sk** paramete is different.
WT: 1e-6
dPBS: 2.6e-7
dOCP: 1.25e-6

However maybe all the experiment data should be normalized to neglect the **sk** parameter.

Other differences is mainly about how to read the file and save the file.

#### Program

### ODE function

ODE_*.py is convenient to compare the model and experiment data like DBsolve. And it will create the plot in **plot** file and model data in **csv** file.


