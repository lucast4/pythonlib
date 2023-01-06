## models for discrete grammar expt
import pandas as pd


# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_data_beh_model_scores(D, list_rules, dict_map_epoch_to_rulenames):
    results = []
    for i in range(0, len(D.Dat)):
        # TODO: this should work for all rules, inclding "rank", "chain", "directions", etc.
        # TODO: handle case for ambiguous (i.e. two prims in same row, or col) — R can be UD not L; U can be RL not D
        print("trial #", i)
        # generate all parses for this trial
        GD = D.grammar_parses_generate(i, list_rules)

        # extract beh sequence
        gramdict = D.sequence_extract_beh_and_task(i, False)
        taskstroke_inds_beh_order = gramdict["taskstroke_inds_beh_order"]
        
        # append to results
        epoch_i = D.Dat.iloc[i]['epoch']
        if epoch_i in dict_map_epoch_to_rulenames.keys():
            epoch_i = dict_map_epoch_to_rulenames[epoch_i]
        
        results.append({'trial_num':i,'epoch':epoch_i,'character':D.Dat.iloc[i]['character'],'trialcode':D.Dat.iloc[i]['trialcode']})

        # for each rule, check if trial succeeds on that rule
        for rule in list_rules:    
            print("rule:", rule)
            # get parses
            parses = GD.ChunksListClassAll[rule].search_permutations_chunks() # combines Hier, Fixed Order into one
            
            # add success value to corresponding column
            results[-1][f"behmodpost_{rule}_default"] = taskstroke_inds_beh_order in parses
    df = pd.DataFrame(results)
#     if dict_map_epoch_to_rulenames: # dict_map_epoch_to_rulenames is NOT empty
#         df = applyFunctionToAllRows(df, mapEpochToRuleNames, 'epoch')
    return df

# modify existing D in-place, adding column 'binary_rule_tuple'
def add_binary_rule_tuple_col(df, rule_cols):
    tuple_col_name = 'binary_rule_tuple'
    df[tuple_col_name] = ''

    for i in range(0,len(df)):
        df.at[i, tuple_col_name] = str(tuple([int(df.at[i, x]) for x in rule_cols]))

