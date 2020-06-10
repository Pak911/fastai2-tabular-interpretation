from pickle import load, dump
from bz2 import BZ2File
import pandas as pd


def sv_var(var, name, path, bzipped=False):
    "Save variable as pickle object to path with name"
    if (bzipped == False):
        f = open(path/f"{name}.pkl","wb")
    else:
        f = BZ2File(path/f"{name}.pkl.bz2", "wb")
    dump(var, f)
    f.close()

def ld_var(name, path, bzipped=False):
    "Returns a pickle object from path with name"
    if (bzipped == False):
        f = open(path/f"{name}.pkl","rb")
    else:
        f = BZ2File(path/f"{name}.pkl.bz2","rb")
    var = load(f)
    f.close()
    return var


def uniqueness(df):
    """
    Shows how many different values each column have
    """
    result = pd.DataFrame(columns=['column', 'uniques', 'uniques %'])
    ln = len(df)
    for col in df:
        uniqs = len(df[col].unique())
        result = result.append({'column':col, 'uniques':uniqs, 'uniques %':uniqs/ln*100}, ignore_index=True)
    return result.sort_values(by='uniques', ascending=False)   



def _list_diff(list_1:list, list_2:list)->list:
    "Difference between first and second lists"
    diff = set(list_1) - set(list_2)
    return [item for item in list_1 if item in diff]

def list_diff(list1, list2, *args)->list:
    "Difference between first and any number of lists"
    diff = _list_diff(list1, list2)
    for arg in args:
        diff = _list_diff(diff, arg)
    return diff

def ifNone(a, b):
    "returns `b` if `a` is None else return `a`"
    return b if isNone(a) else a

def isNone(cond):
    return cond is None

def isNotNone(cond):
    return cond is not None

def listify(p=None, match=None):
    "Make `p` listy and the same length as `match`."
    if p is None: p=[]
    elif isinstance(p, str): p = [p]
    else:
        try: a = len(p)
        except: p = [p]
    n = match if type(match)==int else len(p) if match is None else len(match)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

def is_in_list(values:list, in_list:list)->bool:
    '''
    Just returns is any of the elements from values is in list in_list
    '''
    if (len(which_elms(values, in_list)) > 0):
        return True
    else:
        return False
    
def which_elms(values:list, in_list:list)->list:
    '''
    Just returns elements from values that are in list in_list
    '''
    return [x for x in values if (x in in_list)]