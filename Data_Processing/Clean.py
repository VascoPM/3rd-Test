import pandas as pd

def NaN_policy (df, window_cols, method_fullrows = 'ffill', method_sparserows = 'ffill'):
    # This functions checks and corrects NaNs.
    # It has redundancy built into it to double check for NaNs.
    # And allows for different policies for sparse NaN values and rows full of NaNs.
    
    df = df.copy(deep=True)
        
    if df[window_cols].isna().any().any():
        print('NaN values detected.')
        
        nan_exist = True
        i = 0   
        while nan_exist:
            
            # Makes sure the policy only iterates with one redundant round.
            if i >= 2:
                print('NaN Policy Failed')
                break            

            #Checking if there are rows with only NaNs
            if df[window_cols].isna().all(axis=1).any():
                n_nan_rows = len(df[df[window_cols].isna().all(axis=1)])
                print(f'There are {n_nan_rows} rows of all NaN values.')

                #Correcting only NaN rows 
                df.fillna(method= method_fullrows, axis = 0, inplace = True)
                if ~ df[window_cols].isna().all(axis=1).any():
                    print(f'Rows of NaN corrected')
                    if ~ df[window_cols].isna().any().any():
                        nan_exist = False
                        print(f'NaNs no longer detected.')
            
            #Checking if there are rows with any NaNs in them
            if df[window_cols].isna().any(axis=1).any():
                n_nan_rows = len(df[window_cols].isna().any(axis=1))
                print(f'There are {n_nan_rows} rows with some NaN values.')
                
                # Correcting NaN inside rows 
                df.fillna(method= method_sparserows, axis = 1, inplace = True)
                if ~ df[window_cols].isna().any(axis=1).any():
                    print(f'Rows with some NaN were corrected')
                    if ~ df[window_cols].isna().any().any():
                        nan_exist = False
                        print(f'NaNs no longer detected.')

            i += 1            

    else:
        print('No NaNs detected')
        
    return df