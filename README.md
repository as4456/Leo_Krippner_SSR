# Leo_Krippner_SSR
Please check author’s (Leo Krippner) webpage for original code and documentation:  http://www.rbnz.govt.nz/research_and_publications/research_programme/additional_research/5655249.html
Please Note: The results from the “Comparison of international monetary policy measures” in the above link are currently obtained using the K-ANSM(2) with an estimated lower bound method (i.e folder “C_KANSM2_Estimated_LB” in the original code) and that only has been provided in Python here. 
Data Files
All the yield curve data files (i.e A_Country_All_Data_Bloomberg.xlsm) for the respective countries can be updated by opening the .xlsm files in the folder “Data_Files” in a Bloomberg-enabled computer and then saving them in the format (.xls) and keeping it in the same folder as the main script (AAA_RUN_KANSM2_Est_LB.py)
For initial reproduction of results, sample data files (i.e A_Country_All_Data_Bloomberg.xls) for the respective countries have been provided till November 2015.    

Instructions for generating the results
0. Install pip   (E.g., on Unix run "sudo easy_install pip")
1. Install libraries.  (E.g. run "pip install openpyxl")
2.	Run "python data\_read.py"   This generates yield curve data.   The script generates the spliced yield curve dataset (Govt. data spliced with the OIS data after a specific date) for a respective country (Line 61 in the code) in monthly, weekly and daily csv formats.
3.	Run "python AAA\_RUN\_KANSM2\_Est\_LB.py"  This generates the shadow rate and other results.  The script generates the results in a csv format as in the “Comparison of international monetary policy measures” for a respective country (Line 27) in the desired frequency (Line 28).
Please Note: Currently the code uses given parameters (FinalNaturalParameters_Country.dat) but you have the option (Line 23) of estimating it from the whole dataset, although the code running time becomes slower and needs to be optimized further.
