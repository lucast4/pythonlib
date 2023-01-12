## models for discrete grammar expt
import pandas as pd


# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_scored_beh_model_data(D, list_rules, dict_map_epoch_to_rulenames,binary_rule=False):
    results = [] #dataframe to return

    for i in range(0, len(D.Dat)):
        print("trial #", i)
        # generate all parses for this trial
        GD = D.grammar_parses_generate(i, list_rules)

        # extract beh sequence
        gramdict = D.sequence_extract_beh_and_task(i, False)
        taskstroke_inds_beh_order = gramdict["taskstroke_inds_beh_order"]
        
        # rename value in 'epoch' to rulename, if applicable
        epoch_i = D.Dat.iloc[i]['epoch']
        if epoch_i in dict_map_epoch_to_rulenames.keys():
            epoch_i = dict_map_epoch_to_rulenames[epoch_i]
        
        # add necessary columns to results
        results.append({'trial_num':i,'epoch':epoch_i,'character':D.Dat.iloc[i]['character'],'trialcode':D.Dat.iloc[i]['trialcode']})

        # for each rule, check if trial succeeds on that rule
        for rule in list_rules:    
            print("rule:", rule)
            if rule in ['left', 'right', 'up', 'down']: #direction
                # get parses
                parses = GD.ChunksListClassAll[rule].search_permutations_chunks() # combines Hier, Fixed Order into one
                # add success value to corresponding column
                results[-1][f"behmodpost_{rule}_default"] = taskstroke_inds_beh_order in parses
            
            # NOTE: if using a hyphenated rule, must use this format:
            elif any(e in rule for e in ['rank', 'chain']): # rule in this case is e.g. 'chain-LVI'
                # get parses
                parses = GD.ChunksListClassAll[rule].search_permutations_chunks()
                # add success value to corresponding column
                results[-1][f"behmodpost_{rule}_default"] = taskstroke_inds_beh_order in parses
            
            else:
                assert False, 'invalid rule'
    df = pd.DataFrame(results)
    
    if binary_rule:
        _add_binary_rule_tuple_col(df, list_rules)
    return df

# for BOOLEAN rules only—adds column for each rule with True/False values.
# @noReturn; modifies existing D in-place, adding column 'binary_rule_tuple'
def _add_binary_rule_tuple_col(df, rule_cols):
    tuple_col_name = 'binary_rule_tuple'
    df[tuple_col_name] = ''

    # NOTE: uses "behmodpost_RULE_default" col format—may need to adapt
    for i in range(0,len(df)):
        df.at[i, tuple_col_name] = str(tuple([int(df.at[i, f"behmodpost_{x}_default"]) for x in rule_cols]))

