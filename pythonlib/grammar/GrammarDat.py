""" Single class for analysis of grammar/sequencing for discrete shapes, combining the 
following:
- Generating parses for given (task, rule) combinations
--- Relates to stereotyped parses from Matlab objectclass (grammar plots)
--- Relates to diagnostic model (gridlinecircle)
- Compute diagnostic scores (gridlinecircle), for (beh, model) x (task) x (score_feature)
- Compute performance (fraction correct) as used in things like neuralbiasdir
"""

class GrammarDat(object):
    """
    See docs above.
    """

    def __init__(self, input_data_dict, input_version):
        """ Input data for a single trial, icnlduing the 
        Beh, the Task
        PARAMS;
        - input_data_dict, depends on input_version, see below
        - input_version, str, what kind of data is inputed.
        """

        # Properties
        self.ChunksListClassAll = {} # dict[rulename] = CLC for that rule

        # Input data
        if input_version=="dataset":
            """ Input the dataset and a trial index
            """
            D = input_data_dict["dataset"]
            ind = input_data_dict["ind_dataset"]

            Beh = D.Dat.iloc[ind]["BehClass"]
            Task = D.Dat.iloc[ind]["Task"]
            expt_list = D.expts()
            assert len(expt_list)==1, "modify code to input the expt for this specifig trial..."
            exptname = expt_list[0]
            self._input_data(Beh, Task, exptname)

        elif input_version=="direct":
            # Directly input the Beh and Task
            assert False, "code it"
        else:
            print(input_version)
            print(input_data_dict)
            assert False


    def _input_data(self, Beh, Task, exptname):
        """ Input data. 
        """

        self.Beh = Beh
        self.Task = Task
        self.Expt = exptname


    def parses_generate_batch(self, list_rules):
        """ Generate all parses for this list of rules
        PARAMS;
        - list_rules, lsit of str, each a rule, like lolli
        RETURNS:
        - adds to self.ChunksListClassAll[rule] = parses, where paress
        are list of orderings (chunks)
        """
        for rule in list_rules:
            if rule in self.ChunksListClassAll.keys():
                # Then parses already generated. ignore
                # parses = GD.ChunksListClassAll[rule]
                pass
            else:
                # Then generate, save, and return parses
                self.parses_generate(rule)

    def parses_generate(self, rule_name, DEBUG=False):
        """
        Generate and cache parses for this rule, in form of ChunksListClass
        PARAMS:
        - rule_name, string name of rule that will be applied to generate parses.
        RETURNS:
        """
        from pythonlib.chunks.chunksclass import ChunksClassList
        from pythonlib.behavior.behaviorclass import BehaviorClass

        # motifkind = motifname
        # motif_params = motifparams

        # Extract all chunks consistent with this rule for this task
        # Task = D.Dat.iloc[indtrial]["Task"]
        params = {
            "Task":self.Task,
            "expt":self.Expt, 
            "rule":rule_name
        }
        CL = ChunksClassList(method="task_entry", params=params)
        CL.remove_chunks_that_concat_strokes() # Only keep the chunks that are defined by hierarhcy, not by concatting strokes        
        
        if DEBUG:
            CL.print_summary()
    
        if False:        
            # Actually, is fine to be empty.
            if len(CL.ListChunksClass)==0:
                # Then no chunks exist
                return None, None
        
        # Save it
        self.ChunksListClassAll[rule_name] = CL

        # Genreate all concrete parses and return them
        out = CL.search_permutations_chunks(return_ver="list_of_flattened_chunks")
        if DEBUG:
            print(" == Printing all concrete chunks")
            for o in out:
                print(o)
        print("TODO: confirm that no duplicate chunks")

        return out

    ############### DIAGNOSTIC MODELING



    ############### PLOT
    # Plot all parses for each rule, and compare to the beh drawing.
    
    



