import os
import sys
from argparse import ArgumentParser

cwd = os.getcwd()
modules = cwd.replace('/exe/IF_analysis', '')
sys.path.insert(1, modules)

import modules.configs_setup as configs
import modules.processed_data_analysis as pda
import modules.javaThings as jv
jv.start_jvm()
jv.init_logger()


# parse argument from command line
parser = ArgumentParser()
parser.add_argument('data_path', type=str, help='A required str argument providing the path to a folder with czi files')
args = parser.parse_args()
raw_data_folder = args.data_path

# load cnfgs
cnfgs = configs.load_configs(raw_data_folder)
print(cnfgs)


# load data
results_path = cnfgs['results_folder']
df_path = cnfgs['merged']['save_name']
df_all = pda.load_processed_df(df_path)


###################################################
orientation_channel = 'Bra' # pick channel used to orient the AP axis profile
format = '.pdf'


###################################################
# preprocess data for plotting
# outlier flagging from config dict with key = expID: value = list of outliers style '01'
outlier_dict = cnfgs['outliers']
df_all_flagged = pda.filter_outliers(cnfgs['outliers'], df_all, label='outlier')
cnfgs['outliers_flagged'] = {'flagged': True, 'label': 'outlier'}

# convert units from pixel to metric units
df_all_flagged = pda.convert_units_to_um(df_all_flagged, args=['Length_MA', 'Volume_MA'])
cnfgs['unit_conversion'] = True

# div by area
dict_channel = cnfgs['dict_channel'][0]
df_all_flagged = pda.div_by_area(df_all_flagged, channels=dict_channel.keys())

# determine the AP axis orientation of the profiles given an orientation channel (not fipping them)
oc = 'Bra'
df_all_flagged['correct_orientation'] = pda.get_profile_orientation(df_all_flagged, orientation_channel= oc)
cnfgs['profile_orientation'] = {'orientation_bool': True, 'orientation_channel': oc}

# save data frame
df_all_flagged.to_json(df_path, orient='split')
print('Preprocessed df has been saved under ', df_path)

configs.update_configs(cnfgs)

jv.kill_jvm()
os._exit(1)
