import pandas as pd
import scipy


def Zscore (df_in, window_cols, all_neg_replace = -1):
    # retriving sets of columns
    cols = df_in.columns.to_list()
    other_cols = list(set(cols) - set(window_cols))
    
    # Zscore can't handle constant (flat) inputs
    # Therefore windows with all negative credit output NaNs
    # Instead these are excluded from zcoring and are manually scaled
    # Default scalling is  -7 => -1
    all_neg_replace = float(all_neg_replace)
    all_neg_index = df_in[(df_in[window_cols] == float(-7)).all(axis=1)].index
    positive_index = list(set(df_in.index.to_list()) - set(all_neg_index))
    
    if len(all_neg_index.to_list()) > 0:
        # Aux DF for replacing -7 with all_neg_replace (i.e, -1)
        df_allneg = df_in.iloc[all_neg_index].copy()
        df_allneg.loc[:, window_cols] = all_neg_replace

        # Zscoring only non-constant rows
        df_positive = df_in.iloc[positive_index].copy()    
        df_positive[window_cols] = scipy.stats.zscore(df_positive[window_cols],
                                                         axis=1,
                                                         nan_policy = 'omit')

        df_zscore = pd.concat([df_positive, df_allneg])
        df_zscore.sort_values(by=['short_ID', 'window_ID'], inplace = True)

        # If indeces match, then the join was successful
        if df_zscore.index.to_list() == df_in.index.to_list():
            return df_zscore
        # Otherwise raise error
        else:
            print('Error Zscoring')
    else:
        df_zscore = pd.DataFrame(columns= cols)
        df_zscore[window_cols] = scipy.stats.zscore(df_in[window_cols],
                                                         axis=1,
                                                         nan_policy = 'omit')
        df_zscore[other_cols] = df_in[other_cols]
        

    return df_zscore