

def poster_dataset():
    """ 
    OUT:
    - Scorer
    """    
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.drawmodel.strokedists import distscalarStrokes

    Po = Scorer()
    def F(likelis, priorprobs):
        from pythonlib.behmodel.scorer.utils import posterior_score
        postscore = posterior_score(likelis, priorprobs, "weighted")
        return postscore
    Po.input_score_function(F)
    return Po
