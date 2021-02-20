if select_col:
#     h = pd.read_csv(read_path)
#     hold.insert(0,'Y',h['Y'])
#     df = pd.read_csv(f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\{dataIS}_Data\\VP001-{dataIS}\\0-back_session\\data_0-back_session_trial001.csv')
#     col_list = []
#     c = 0
#     for col in df.columns:
#         if 'S1D1'in col or 'S13D13' in col:
#             if 'HbT' in col:
#                 c+=1
#             else:
#                 hold.insert(len(hold.columns),df.columns[c],h[f'{c}'])
#                 c +=1
#         else:
#             c+=1
# else:
#     hold = pd.read_csv(read_path)