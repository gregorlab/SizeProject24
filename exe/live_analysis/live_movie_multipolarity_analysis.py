import os
import time
from sys import path
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/live_analysis', '')
path.insert(1, modules)

import modules.configs_setup as configs
import modules.raw_live_analysis as raw
import modules.processed_data_analysis as pda

# parse argument from command line
parser = ArgumentParser()
parser.add_argument('data_path', type=str, help='A required str argument providing the path to a folder with movie raw files')
parser.add_argument('condition', type=str, help='Specify condition to analyze: "150", "300", "600", "1200", or "all".')
args = parser.parse_args()

data_folder = args.data_path
cond = args.condition

# load configs
cnfgs = configs.load_configs(data_folder)
print(cnfgs.keys())


if cond == 'all':
    folder_OIs = [c for c in cnfgs['data_paths'].keys() if not c == 'ending']
else:
    folder_OIs = [cond]
print('Analyzing conditions ', folder_OIs)


# iterate over folders
for folder_OI in folder_OIs:

    # load lists of paths to dfs and movies
    _, df_list = pda.get_file_list(cnfgs['data_paths'][folder_OI], ending=cnfgs['data_paths']['ending'])
    
    _, movie_list = pda.get_file_list(cnfgs['movie_paths'][folder_OI], ending=cnfgs['movie_paths']['ending'])
    
    print('Starting analysis for condition ', folder_OI)
    start_group_time = time.time()
    
    # start multipolarity analysis
    raw.compute_multipol_analysis_per_condition(df_list, movie_list, ch_fluo=2, k=7, th=70)

    print('Analyis completed for ', folder_OI)

    group_time = time.time()
    print(f" analysis time for group {folder_OI} --- %s seconds ---" % (group_time - start_group_time))

os._exit(1)