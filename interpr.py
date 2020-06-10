__all__ = ['Dendrogramm', 'PermutFI', 'InterpretWaterfall', 'PartDep', 'EmbeddingsViz']

from IPython.display import clear_output
from plotnine import *
import plotly.graph_objects as go
import ast
from scipy.cluster import hierarchy as hc
from sklearn import manifold
from fastai2.imports import *
from fastai2.tabular.all import *
from utils import *



class Interpret():
    def __init__(self, learn, df):
        """
        MasterClass what knows how to deal with learner and dataframe
        Now for classification only
        """
        self.learn = learn
        self.df = df

    def _predict_row(self, row):
        """
        Wrapper for prediction on a single row
        """
        learn = self.learn
        return float(learn.get_preds(dl=learn.dls.test_dl(pd.DataFrame([row])))[0][0][0])

    def _predict_df(self, df=None, is_ret_actls=False):
        """
        returns predictions of df with certain learner
        """
        df = df if isNotNone(df) else self.df
        if (is_ret_actls == False):
            return np.array(self.learn.get_preds(dl=self.learn.dls.test_dl(df))[0].T[0])
        else:
            out = self.learn.get_preds(dl=self.learn.dls.test_dl(df))
            return np.array(out[0].T[0]), np.array(out[1].T[0])

    def _convert_dep_col(self, dep_col, use_log=False):
        '''
        Converts dataframe column, named "depended column", into tensor, that can later be used to compare with predictions.
        Log will be applied if use_log is set True
        '''
        actls = self.df[dep_col].T.to_numpy()[np.newaxis].T.astype('float32')
        actls = np.log(actls) if (use_log == True) else actls
        return torch.tensor(actls)

    def _list_to_key(self, field):
        """
        Turns unhashable list of strings to hashable key
        """
        return f"{field}" if isinstance(field, str) else ', '.join(f"{e}" for e in field)

    def _sv_var(self, var, name, path: Path = None):
        "Save variable as pickle object to path with name"
        f = open(path / f"{name}.pkl", "wb")
        dump(var, f)
        f.close()

    def _ld_var(self, name, path: Path = None):
        "Returns a pickle object from path with name"

        f = open(path / f"{name}.pkl", "rb")
        var = load(f)
        f.close()
        return var

    def _calc_loss(self, pred, targ):
        '''
        Calculates error from predictions and actuals with a learner loss function
        '''
        func = self.learn.loss_func
        return func(torch.tensor(pred, device=default_device()), torch.tensor(targ, device=default_device()))

    def _calc_error(self, df=None):
        '''
        Wrapping function to calculate error for new dataframe on existing learner (learn.model)
        See following functions' docstrings for details
        '''
        df = df if isNotNone(df) else self.df
        preds, actls = self._predict_df(df=df, is_ret_actls=True)
        error = self._calc_loss(pred=preds, targ=actls)
        return float(error)

    def _get_cat_columns(self, is_wo_na=False):
        if (is_wo_na == False):
            return self.learn.dls.cat_names
        else:
            return self.learn.dls.cat_names.filter(lambda x: x[-3:] != "_na")

    def _get_cont_columns(self):
        return self.learn.dls.cont_names

    def _get_all_columns(self):
        return self._get_cat_columns() + self._get_cont_columns()

    def _get_dep_var(self):
        return self.learn.dls.y_names[0]
    
    
    
class Dendrogramm():
    def __init__(self, df):
        """
        Analize dataframe to build and plot correlation matrix
        """
        self.df = df
        self.corrM = None
        self.corrM = self._get_cramer_v_matr()
    
    def _cramers_corrected_stat(self, confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher,
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        try:
            chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
        except:
            return 0.0
    
        if (chi2 == 0):
            return 0.0
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    
    def _get_cramer_v_matr(self):
        '''
        Calculate Cramers V statistic for every pair in df's columns
        '''
        df = self.df
        cols = list(df.columns)
        corrM = np.zeros((len(cols), len(cols)))
        for col1, col2 in progress_bar(list(itertools.combinations(cols, 2))):
            idx1, idx2 = cols.index(col1), cols.index(col2)
            corrM[idx1, idx2] = self._cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
            corrM[idx2, idx1] = corrM[idx1, idx2]
        np.fill_diagonal(corrM, 1.0)
        return corrM
    
    def _get_top_corr_df(self, corr_thr: float = 0.8):
        df, corr_matr = self.df, self.corrM
        corr = np.where(abs(corr_matr) < corr_thr, 0, corr_matr)
        idxs = []
        for i in range(corr.shape[0]):
            if (corr[i, :].sum() + corr[:, i].sum() > 2):
                idxs.append(i)
        cols = df.columns[idxs]
        return pd.DataFrame(corr[np.ix_(idxs, idxs)], columns=cols, index=cols)
    
    def _get_top_corr_dict_corrs(self, top_corrs):
        cols = top_corrs.columns
        top_corrs_np = top_corrs.to_numpy()
        corr_dict = {}
        for i in range(top_corrs_np.shape[0]):
            for j in range(i + 1, top_corrs_np.shape[0]):
                if (top_corrs_np[i, j] > 0):
                    corr_dict[cols[i] + ' vs ' + cols[j]] = np.round(top_corrs_np[i, j], 3)
        return collections.OrderedDict(sorted(corr_dict.items(), key=lambda kv: abs(kv[1]), reverse=True))
    
    def get_top_corr_dict(self, corr_thr: float = 0.8):
        '''
        Outputs top pairs of correlation in a given dataframe with a given correlation matrix
        Filters output mith minimal correlation of corr_thr
        '''
        top_corrs = self._get_top_corr_df(corr_thr=corr_thr)
        return self._get_top_corr_dict_corrs(top_corrs)
    
    def plot_dendrogram(self, figsize=None, leaf_font_size=16):
        '''
        Plots dendrogram for a calculated correlation matrix
        '''
        corr_matr, columns = self.corrM, self.df.columns
        if (figsize is None):
            figsize = (15, 0.02 * leaf_font_size * len(columns))
        corr_condensed = hc.distance.squareform(1 - corr_matr)
        z = hc.linkage(corr_condensed, method='average')
        fig = plt.figure(figsize=figsize)
        dendrogram = hc.dendrogram(z, labels=columns, orientation='left', leaf_font_size=leaf_font_size)
        plt.show()
    
    def uniqueness(self):
        """
        Shows how many different values each column has
        """
        df = self.df
        result = pd.DataFrame(columns=['column', 'uniques', 'uniques %'])
        ln = len(df)
        for col in df:
            uniqs = len(df[col].unique())
            result = result.append({'column': col, 'uniques': uniqs, 'uniques %': uniqs / ln * 100},
                                   ignore_index=True)
        return result.sort_values(by='uniques', ascending=False)
    
    
    
    
class PermutFI(Interpret):
    def __init__(self, learn, df, rounds=5, fields=None, is_use_cache=False):
        """
        Calculate feature importances for the `fields`
        :param rounds: number of copies of shuffled data, the more rounds the less random result is
        :param fields: list of lists of columns to analyze, connected columns should be in the same list element
        (as a list)
        :param is_use_cache: if True tries to load previous cached result if exist (checks only file existence)
        """
        super().__init__(learn, df)
        self.rounds = rounds
        self.col_names = self._get_all_columns()
        self.fields = fields
        self.is_use_cache = is_use_cache
        self.cache_path = Path(learn.path / 'cache')

        self.fi = self._load_or_calculate()
        self._save_to_cache()

    @classmethod
    def empty_cache(self, learn):
        """
        deletes the cache file
        """
        path = Path(learn.path / 'cache')
        name = 'fi_cache'

        file = Path(f"{path / name}.pkl")

        if not (file.exists()):
            print(f"No chache file {file}")
        else:
            file.unlink()

    def _load_or_calculate(self):
        """
        Calculates fi or load it from cache if possible
        """
        if (self.is_use_cache == False) or isNone(self._load_cached()):
            return self._calc_perm_feat_importance()
        else:
            return self._load_cached()

    def _load_cached(self):
        """
        Load calculated Feature Importance.
        """
        name = 'fi_cache'
        path = self.cache_path

        if not (Path(f"{path / name}.pkl").exists()):
            return None

        return self._ld_var(name=name, path=path)

    def _save_to_cache(self):
        """
        Save calculated Feature importance
        """
        name = 'fi_cache'
        path = self.cache_path
        path.mkdir(parents=True, exist_ok=True)

        self._sv_var(var=self.fi, name=name, path=path)

    def _calc_error_mixed_col(self, sampl_col):
        df, rounds = self.df, self.rounds
        df_temp = pd.concat([df] * rounds, ignore_index=True).copy()
        df_temp[sampl_col] = np.random.permutation(df_temp[sampl_col].values)
        return self._calc_error(df=df_temp)

    def _calc_perm_feat_importance(self):
        """
        Calcutate permutation feature importance for a list of fields
        """

        fields = ifNone(self.fields, self.col_names)
        base_error = self._calc_error()
        importance = {}

        for field in progress_bar(fields):
            key = self._list_to_key(field=field)
            importance[key] = self._calc_error_mixed_col(sampl_col=field)
        clear_output()
        for key, value in importance.items():
            importance[key] = (value - base_error) / base_error
        return collections.OrderedDict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    def plot_importance(self, limit=20, asc=False):
        df_copy = pd.DataFrame([[k, v] for k, v in self.fi.items()], columns=['feature', 'importance']).copy()
        df_copy['feature'] = df_copy['feature'].str.slice(0, 25)
        ax = df_copy.sort_values(by='importance', ascending=asc)[:limit].sort_values(by='importance',
                                                                                     ascending=not (
                                                                                         asc)).plot.barh(
            x="feature", y="importance", sort_columns=True, figsize=(10, 10))
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y() * 1.005))

    def get_least_important(self):
        return OrderedDict(reversed(list(self.fi.items())))
    
    
    
    
class InterpretWaterfall(Interpret):
    def __init__(self, learn, df, fields, sampl_row, max_row_used=None, use_log=False, use_int=False, num_tests=1):
        """
        Calculate all the parameters to plot Waterfall graph for a `sampl_row`
        fields -- list of lists of columns to analyze, connected columns should be in the same list element (as a list)

        max_row_used -- how many rows to use for calculation. len(df) -- by default
            Can be absolute value or coeffficient (from the len(df))
            On big datasets can easily be set to lower values as it's enough data for calculating
            differences anyway. 10k rows is often enough
        num_tests -- id used to reduce memory consumption, each run uses `max_row_used/num_tests` rows, the more
            'num_tests' the less memory consumption is
        use_log=True is needed if we have transformed depended variable into log
        use_int=True is needed if we want to log-detransformed (exponented) var to me integer not float
        """

        super().__init__(learn, df)
        self.fields = fields
        self.sampl_row = sampl_row
        self.use_log = use_log
        self.use_int = use_int
        self.num_tests = num_tests

        if isNone(max_row_used) or (max_row_used > len(df)):
            self.max_row_used = int(len(df) / num_tests)
        elif (max_row_used < 1):
            self.max_row_used = int(len(df) * max_row_used / num_tests)
        else:
            self.max_row_used = int(max_row_used / num_tests)

        print("hold on...")
        self.model_mean = np.array(self.learn.get_preds(dl=self.learn.dls.test_dl(df))[0].T[0]).mean()
        clear_output()
        self.actual = self._predict_row(row=sampl_row)
        self.forces = None
        self.batch_forces = None

        self._calc_forces_repeats()

    def _shuffle_cols(self, sampl_col):
        """
        Returns all the variations of sampl_col columns for a particular row sampl_row.
        (what would be if he had all other values in sampl_col columns)
        Copy all the columns except sampl_col from sampl_row max_row_used times.
        Then add random sampl_col from original distribution
        max_row_used can be < 1, in that case it's a portion of len(df)
        """
        df, learn, sampl_row, max_row_used = self.df, self.learn, self.sampl_row, self.max_row_used

        sampl_col = listify(sampl_col)
        rows = [sampl_row.to_dict()] * max_row_used  # performance optimization
        temp_df = pd.DataFrame(rows)
        shfl_cols = df[sampl_col].sample(max_row_used).copy()
        temp_df[sampl_col] = shfl_cols.values
        return temp_df

    def _calc_forces(self):
        """
        Calculate ordered dict with forces created by certain feature values for particular row
        :return:
        ordered dict of sorted forces
        """
        df, learn, sampl_row, model_mean, fields, max_row_used = self.df, self.learn, self.sampl_row, self.model_mean, self.fields, self.max_row_used

        forces = OrderedDict()
        # build big table with all the variants to check all the data in one run
        huge_df = pd.DataFrame()
        cur_dfs = []
        for field in fields:
            cur_df = self._shuffle_cols(sampl_col=field)
            cur_df['group'] = self._list_to_key(field)
            cur_dfs.append(cur_df)

        huge_df = pd.concat(cur_dfs)
        del cur_dfs

        # predict on it
        huge_df['preds'] = self._predict_df(df=huge_df)
        # predict actual data (just predict sampl_row)
        actual = self._predict_row(row=sampl_row)

        # divide back by fields
        for field in fields:
            cur_df = huge_df.query(f"group == '{self._list_to_key(field)}'")
            # calculate force
            force = float(actual - cur_df['preds'].mean())
            key = f"{field} ({sampl_row[field]})" if isinstance(field, str) else ', '.join(
                f"{e} ({sampl_row[e]})" for e in field)
            forces[key] = force

        self.batch_forces = OrderedDict(sorted(forces.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def _calc_forces_repeats(self):
        """
        Repeat _calc_forces to avg the data and save memory
        """
        num_tests = self.num_tests
        forces = pd.DataFrame()
        for tests in progress_bar(range(num_tests)):
            self._calc_forces()
            forces = forces.append(self.batch_forces, ignore_index=True)
        clear_output()

        forces = forces.mean()
        fc_od = OrderedDict()
        for k, v in forces.iteritems():
            fc_od[k] = v

        self.forces = OrderedDict(sorted(fc_od.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def _conv_exp(self, value, use_log=False, use_int=False):
        """
        Use exponent and convert to integer if needed
        """
        ret_val = value if (use_log == False) else np.exp(value)
        np_int = int if np.isscalar(ret_val) else np.vectorize(np.int)
        ret_val = ret_val if (use_int == False) else np_int(ret_val)
        return ret_val
    
    def _explain_forces(self, forces_show=10):
        """
        :return:
        explained diff (remember 0 expl_diff doent mean no expl in you have 2 explanations with + and -)
        unknown_diff
        """

        def conv_perc(val):
            return (val - 1) * 100

        forces, model_mean, actual, use_log, use_int = self.forces, self.model_mean, self.actual, self.use_log, self.use_int
        expl_diff = np.array(list(forces.values())).sum()
        unk_diff = actual - (model_mean + expl_diff)

        i = 0
        expl_df = pd.DataFrame(columns=['feature', 'coef', 'diff_perc', 'overall_diff', 'overall'])
        # unexplainable value of diff in prices, used in waterfall
        last_price = 0
        price = self._conv_exp(model_mean, use_log=use_log, use_int=use_int)
        expl_df = expl_df.append(pd.DataFrame(
            {'feature': 'model_mean', 'coef': 0, 'diff_perc': 0, 'overall_diff': price - last_price, 'overall': price},
            index=[len(expl_df)]))
        last_price = price
        for key, value in forces.items():
            if (i >= forces_show):
                break
            coef = self._conv_exp(value, use_log=use_log)
            price = int(price * coef)
            diff = conv_perc(coef)
            expl_df = expl_df.append(pd.DataFrame(
                {'feature': key, 'coef': coef, 'diff_perc': diff, 'overall_diff': price - last_price, 'overall': price},
                index=[len(expl_df)]))
            last_price = price
            i += 1
        actl_price = self._conv_exp(actual, use_log=use_log, use_int=use_int)
        coef = actl_price / price
        diff = conv_perc(coef)
        expl_df = expl_df.append(
            pd.DataFrame(
                {
                    'feature': 'others and interconnections', 'coef': coef, 'diff_perc': diff,
                    'overall_diff': actl_price - last_price, 'overall': actl_price
                }, index=[len(expl_df)]
            )
        )

        return expl_df

    def _plot_force_df(self, force_df, name=None):
        height = max(600, int(len(force_df) / 3 * 100))

        measure = ["relative"] * len(force_df)
        x, text, y = [], [], []
        for i, row in force_df.iterrows():
            y.append(row['feature'][:100])
            text.append(f"{row['overall']} ({row['diff_perc']:+.2f}%)")
            x.append(row['diff_perc'])

        fig = go.Figure(go.Waterfall(
            name="20", orientation="h",
            measure=["relative"] * len(force_df),
            x=x,
            textposition="outside",
            text=text,
            y=y,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        title = "Analysis " if isNone(name) else f"Analysis of {name}"
        fig.update_layout(
            title=title,
            showlegend=False,
            height=height,
        )
        fig.update_yaxes(showticklabels=False)

        return fig

    def get_forces_fig(self, name=None, forces_show=20):
        """
        Get the force field calculated earlier as plotly figure
        """
        df = self._explain_forces(forces_show=forces_show)
        return self._plot_force_df(force_df=df, name=name)

    def plot_forces(self, name=None, forces_show=20):
        """
        Plot the force field calculated earlier as dynamic json-frame
        """
        fig = self.get_forces_fig(name=name, forces_show=forces_show)
        fig.show()

    def get_forces(self):
        """
        Returns forces as an Ordered Dict
        """
        return self.forces
    
    def get_variants_pd(self, sampl_col, fields):
        """
        returns df with all the values of dep_var for every variant of fields
        helps to determine best value of fields for particular sampl_col
        """
        use_log, use_int = self.use_log, self.use_int
        uniqs, variants = self._uniq_cols(sampl_col=sampl_col, fields=fields)
        preds = self._predict_df(df=variants)
        dep_var = self._get_dep_var()
        uniqs[dep_var] = self._conv_exp(value=preds, use_log=use_log, use_int=use_int)
        result = pd.DataFrame([], columns=['feature', dep_var, 'times'])
        for i, row in uniqs.iterrows():
            feats = self._list_to_key(list(row[fields]))
            result = result.append({'feature':feats, dep_var:row[dep_var], 'times':row['counts']}, ignore_index=True)
        return result.sort_values(by=dep_var, ascending=False)
        
    
    def _uniq_cols(self, sampl_col, fields):
        """
        Returns df with all unique values in 'fields' other values are copied from 'sampl_col'
        """
        df, learn, sampl_row, max_row_used = self.df, self.learn, self.sampl_row, self.max_row_used
        max_rows = int(self.max_row_used*self.num_tests)
        
        sampl_col = listify(sampl_col)
        uniqs = df.groupby(fields).size().reset_index(name='counts')
        uniqs = uniqs.sort_values(by='counts', ascending=False)
        uniqs = uniqs.head(max_rows)
        
        temp_df = pd.DataFrame([sampl_row.to_dict()] * len(uniqs))
        temp_df[fields] = uniqs[fields].values
        return uniqs, temp_df
    
    def plot_variants(self, sampl_col, fields, limit=20, asc=False):
        """
        Plots how every variant of fields influences dep_var for a particular sampl_col
        helps to determine best value of fields for particular sampl_col
        """

        def prepare_colors(df_pd: pd.DataFrame):
            heat_min = df_pd['times'].min()
            heat_max = df_pd['times'].max()
            dif = heat_max - heat_min
            colors = [((times - heat_min) / (dif), (times - heat_min) / (4 * dif), 0.75) for times in df_pd['times']]
            return colors
        use_int = self.use_int
        df = self.get_variants_pd(sampl_col=sampl_col, fields=fields)
        dep_var = self._get_dep_var()

        df['feature'] = df['feature'].str.slice(0, 45)
        df = df.sort_values(by=dep_var, ascending=asc)[:limit].sort_values(by=dep_var, ascending=not (asc))
        colors = prepare_colors(df_pd=df)
        ax = df.plot.barh(x="feature", y=dep_var, sort_columns=True, figsize=(10, 10), color=colors)
        ax.set_ylabel(fields)
        for (p, t) in zip(ax.patches, df['times']):
            frmt = f'{p.get_width():.0f}' if (use_int == True) else f'{p.get_width():.4f}' 
            ax.annotate(frmt, ((p.get_width() * 1.005), p.get_y() * 1.005))
            ax.annotate(f'{int(t)}', ((p.get_width() * .45), p.get_y() + 0.1), color='white', weight='bold')
            
            
            
            
class PartDep(Interpret):
    """
    Calculate Partial Dependence. Countinious vars are divided into buckets and are analized as well
    Fields is a list of lists of what columns we want to test. The inner items are treated as connected fields.
    For ex. fields = [['Store','StoreType']] mean that Store and StoreType is treated as one entity
    (it's values are substitute as a pair, not as separate values)
    coef is useful when we don't want to deal with all the variants, but only with most common
    In short if coef for ex. is 0.9, then function outputs number of occurrences for all but least 10%
    of the least used
    If coef is more 1.0, then 'coef' itself is used as threshold (as min number of occurances)
    use_log=True is needed if we have transformed depended variable into log
    use_int=True is needed if we want to log-detransformed (exponented) var to me integer not float
    is_couninue=True helps with long calculation, it continues the last calculation from the saved file
    is_use_cache=True loads last fully calculated result. Can distinct caches that were mede with different
    fields and coef
    no_precalc=True -- don't calculate PartDep (usefull if you want to use `plot_raw` and `plot_model` only)
    """

    def __init__(self, learn, df, model_name: str, fields: list = (), coef: float = 1.0,
                 is_sorted: bool = True, use_log=False, use_int=False,
                 cache_path=None, is_use_cache=True, is_continue=False, no_precalc=False):
        super().__init__(learn, df)
        self.use_log = use_log
        self.use_int = use_int
        self.coef = coef
        self.is_sorted = is_sorted

        if (fields is None) or (len(fields) == 0):
            self.fields = self._get_all_columns()
        else:
            self.fields = listify(fields)

        self.part_dep_df = None
        self.cache_path = ifnone(cache_path, learn.path / 'cache')
        self.save_name = f"{model_name}_part_dep"
        self.is_use_cache = is_use_cache
        self.is_continue = is_continue
        self.dep_var = self._get_dep_var()
        if (no_precalc==False):
            self._load_or_calculate()

    @classmethod
    def what_cached(self, model_name: str, path=None, learn=None):
        """
        Shows what keys are cached
        """
        if isNone(path) and isNone(learn):
            print("path and learn cannot be None at the same time")
            return
        elif isNone(path):
            path = learn.path

        name = f"{model_name}_part_dep"
        folder = 'cache'
        path = path / folder

        if not (Path(f"{path / name}.pkl").exists()):
            print(f"No chache file")
        else:
            f = open(path / f"{name}.pkl", "rb")
            var = load(f)
            f.close()
            for k in var.keys():
                print(k)

    @classmethod
    def empty_cache(self, model_name: str, path=None, learn=None):
        """
        deletes the cache file
        """
        if isNone(path) and isNone(learn):
            print("path and learn cannot be None at the same time")
            return
        elif isNone(path):
            path = learn.path

        name = f"{model_name}_part_dep"
        folder = 'cache'
        path = path / folder

        files = (Path(f"{path / name}.pkl"), Path(path / 'pd_interm.pkl'))

        for file in files:
            if not (file.exists()):
                print(f"No chache file {file}")
            else:
                file.unlink()

    def _cont_into_buckets(self, df_init, CONT_COLS):
        """
        Categorical values can be easily distiguished one from another
        But that doesn't work with continious values, we have to divede it's
        values into buckets and then use all values in a bucket as a single value
        that avarages the bucket. This way we convert cont feture into pseudo categorical
        and are able to apply partial dependense analysis to it
        """
        fields = self.fields
        df = df_init.copy()
        if is_in_list(values=fields, in_list=CONT_COLS):
            for col in which_elms(values=fields, in_list=CONT_COLS):
                edges = np.histogram_bin_edges(a=df[col].dropna(), bins='auto')
                for x, y in zip(edges[::], edges[1::]):
                    df.loc[(df[col] > x) & (df[col] < y), col] = (x + y) / 2
        return df

    def _get_field_uniq_x_coef(self, df: pd.DataFrame, fields: list, coef: float) -> list:
        '''
        This function outputs threshold to number of occurrences different variants of list of columns (fields)
        In short if coef for ex. is 0.9, then function outputs number of occurrences for all but least 10%
        of the least used
        If coef is more 1.0, then 'coef' itself is used as threshold
        '''
        if (coef > 1):
            return math.ceil(coef)
        coef = 0. if (coef < 0) else coef
        occs = df.groupby(fields).size().reset_index(name="Times").sort_values(['Times'], ascending=False)
        num = math.ceil(coef * len(occs))
        if (num <= 0):
            # number of occurances is now = max_occs+1 (so it will be no items with this filter)
            return occs.iloc[0]['Times'] + 1
        else:
            return occs.iloc[num - 1]['Times']

    def _get_part_dep_one(self, fields: list, masterbar=None) -> pd.DataFrame:
        '''
        Function calculate partial dependency for column in fields.
        Fields is a list of lists of what columns we want to test. The inner items are treated as connected fields.
        For ex. fields = [['Store','StoreType']] mean that Store and StoreType is treated as one entity
        (it's values are substitute as a pair, not as separate values)
        coef is useful when we don't want to deal with all the variants, but only with most common
        '''
        NAN_SUBST = '###na###'
        cont_vars = self._get_cont_columns()
        fields = listify(fields)
        coef, is_sorted, use_log, use_int = self.coef, self.is_sorted, self.use_log, self.use_int
        dep_name = self._get_dep_var()

        df = self._cont_into_buckets(df_init=self.df, CONT_COLS=cont_vars)

        # here we prepare data to eliminate pairs that occure too little
        # and make NaN a separate value to appear in occures
        field_min_occ = self._get_field_uniq_x_coef(df=df, fields=fields, coef=coef)
        df[fields] = df[fields].fillna(NAN_SUBST)  # to treat None as a separate field
        occs = df.groupby(fields).size().reset_index(name="Times").sort_values(['Times'], ascending=False)
        occs[fields] = occs[fields].replace(to_replace=NAN_SUBST, value=np.nan)  # get back Nones from NAN_SUBST
        df[fields] = df[fields].replace(to_replace=NAN_SUBST, value=np.nan)  # get back Nones from NAN_SUBST
        occs = occs[occs['Times'] >= field_min_occ]
        df_copy = df.merge(occs[fields]).copy()

        # here for every pair of values of fields we substitute it's values in original df
        # with the current one and calculate predictions
        # So we predict mean dep_var for every pairs of value of fields on the whole dataset
        frame = []
        ln = len(occs)
        if (ln > 0):
            for _, row in progress_bar(occs.iterrows(), total=ln, parent=masterbar):
                # We don't need to do df_copy = df.merge(occs[field]).copy() every time
                # as every time we change the same column (set of columns)
                record = []
                for fld in fields:
                    df_copy[fld] = row[fld]
                preds = self._predict_df(df=df_copy)
                preds = np.exp(np.mean(preds)) if (use_log == True) else np.mean(preds)
                preds = int(preds) if (use_int == True) else preds
                for fld in fields:
                    record.append(row[fld])
                record.append(preds)
                record.append(row['Times'])
                frame.append(record)
        # Here for every pair of fields we calculate mean dep_var deviation
        # This devition is the score that shows how and where this partucular pair of fields
        # moves depend valiable
        # Added times to more easily understand the data (more times more sure we are)
        out = pd.DataFrame(frame, columns=fields + [dep_name, 'times'])
        median = out[dep_name].median()
        out[dep_name] /= median
        if (is_sorted == True):
            out = out.sort_values(by=dep_name, ascending=False)
        return out

    def _get_part_dep(self):
        '''
        Makes a datafreme with partial dependencies for every pair of columns in fields
        '''
        fields = self.fields
        learn = self.learn
        cache_path = self.cache_path
        dep_name = self._get_dep_var()
        is_continue = self.is_continue
        l2k = self._list_to_key
        result = []
        to_save = {}
        from_saved = {}

        # Load from cache
        if (is_continue == True):
            if Path(cache_path / 'pd_interm.pkl').exists():
                from_saved = ld_var(name='pd_interm', path=cache_path)
            else:
                is_continue = False

        elapsed = []
        left = []
        if (is_continue == True):
            for field in fields:
                if (l2k(field) in from_saved):
                    elapsed.append(field)
                    new_df = from_saved[l2k(field)]
                    result.append(new_df)
                    to_save[l2k(field)] = new_df

        for field in fields:
            if (l2k(field) not in from_saved):
                left.append(field)

        # Calculate
        pbar = master_bar(left)
        sv_var(var=to_save, name='pd_interm', path=cache_path)
        for field in pbar:
            new_df = self._get_part_dep_one(fields=field, masterbar=pbar)
            new_df['feature'] = self._list_to_key(field)
            if is_listy(field):
                new_df['value'] = new_df[field].values.tolist()
                new_df.drop(columns=field, inplace=True)
            else:
                new_df = new_df.rename(index=str, columns={str(field): "value"})
            result.append(new_df)
            to_save[l2k(field)] = new_df
            sv_var(var=to_save, name='pd_interm', path=cache_path)
        clear_output()
        if Path(cache_path / 'pd_interm.pkl').exists():
            Path(cache_path / 'pd_interm.pkl').unlink()  # delete intermediate file
        result = pd.concat(result, ignore_index=True, sort=True)
        result = result[['feature', 'value', dep_name, 'times']]
        clear_output()

        self.part_dep_df = result

    def _load_dict(self, name, path):
        if not (Path(f"{path / name}.pkl").exists()):
            return None
        return self._ld_var(name=name, path=path)

    def _save_cached(self):
        """
        Saves calculated PartDep df into path.
        Can be saved more than one with as an dict with fields as key
        """

        path = self.cache_path
        path.mkdir(parents=True, exist_ok=True)
        name = self.save_name

        sv_dict = self._load_dict(name=name, path=path)
        key = self._list_to_key(self.fields + [self.coef])
        if isNone(sv_dict):
            sv_dict = {key: self.part_dep_df}
        else:
            sv_dict[key] = self.part_dep_df

        self._sv_var(var=sv_dict, name=name, path=path)

    def _load_cached(self):
        """
        Load calculated PartDep df if hash exist.
        """
        name = self.save_name
        path = self.cache_path

        if not (Path(f"{path / name}.pkl").exists()):
            return None

        ld_dict = self._ld_var(name=name, path=path)
        key = self._list_to_key(self.fields + [self.coef])
        if (key not in ld_dict):
            return None

        return ld_dict[key]

    def _load_or_calculate(self):
        """
        Calculates part dep or load it from cache if possible
        """
        if (self.is_use_cache == False) or isNone(self._load_cached()):
            self._get_part_dep()
            return self._save_cached()
        else:
            self.part_dep_df = self._load_cached()
            
    def _general2partial(self, df):
        if (len(df) == 0):
            return None
        copy_df = df.copy()
        feature = copy_df['feature'].iloc[0]
        copy_df.drop(columns='feature', inplace=True)
        copy_df.rename(columns={"value": feature}, inplace=True)
        return copy_df
        

    def plot_raw(self, field, sample=1.0):
        """
        Plot dependency graph from data itself
        field must be list of exactly one feature
        sample is a coef to len(df). Lower if kernel use to shut down on that
        """
        df = self.df
        df = df.sample(int(len(df)*sample))
        field = field[0]
        dep_var = f"{self._get_dep_var()}_orig"
        return ggplot(df, aes(field, dep_var)) + stat_smooth(se=True, method='loess');

    def plot_model(self, field, strict_recalc=False, sample=1.0):
        '''
        Plot dependency graph from the model.
        It also take into account times, so plot becomes much more resilient, cause not every value treats as equal
        (more occurences means more power)
        field must be list of exactly one feature
        strict_recalc=True ignores precalculated `part_dep_df` and calculate it anyway
        sample is a coef to len(df). Lower if kernel use to shut down on that
        '''
        cached = self.get_pd(feature=self._list_to_key(field))
        
        if (strict_recalc == False) and isNotNone(cached):
            pd_table = cached
        else:
            pd_table = self._get_part_dep_one(fields=field)
            
        clear_output()
        field = field[0]
        dep_var = f"{self._get_dep_var()}"
        rearr = []
        for var, fee, times in zip(pd_table[field], pd_table[dep_var], pd_table['times']):
            for i in range(int(times)):
                rearr.append([var, fee])
        rearr = pd.DataFrame(rearr, columns=[field, dep_var])
        
        rearr = rearr.sample(int(len(rearr)*sample))
        return ggplot(rearr, aes(field, dep_var)) + stat_smooth(se=True, method='loess');

    def get_pd(self, feature, min_tm=1):
        """
        Gets particular feature subtable from the whole one (min times is optional parameter)
        """
        df =  self.part_dep_df.query(f"""(feature == "{feature}") and (times > {min_tm})""")
        return self._general2partial(df=df)

    def get_pd_main_chained_feat(self, main_feat_idx=0, show_min=1):
        """
        Transforms whole features table to get_part_dep_one output table format
        """

        def get_xth_el(str_list: str, indexes: list):
            lst = str_list if is_listy(str_list) else ast.literal_eval(str_list)
            lst = listify(lst)
            if (len(lst) == 1):
                return lst[0]
            elif (len(lst) > 1):
                if (len(indexes) == 1):
                    return lst[indexes[0]]
                else:
                    return [lst[idx] for idx in indexes]
            else:
                return None

        feat_table = self.part_dep_df

        main_feat_idx = listify(main_feat_idx)
        feat_table_copy = feat_table.copy()
        func = functools.partial(get_xth_el, indexes=main_feat_idx)
        feat_table_copy['value'] = feat_table_copy['value'].apply(func)
        feat_table_copy.drop(columns='feature', inplace=True)
        return feat_table_copy.query(f'times > {show_min}')

    def plot_part_dep(self, fields, limit=20, asc=False):
        """
        Plots partial dependency plot for sublist of connected `fields`
        `fields` must be sublist of `fields` given on initalization calculation
        """

        def prepare_colors(df_pd: pd.DataFrame):
            heat_min = df_pd['times'].min()
            heat_max = df_pd['times'].max()
            dif = heat_max - heat_min
            colors = [((times - heat_min) / (dif), (times - heat_min) / (4 * dif), 0.75) for times in df_pd['times']]
            return colors

        df = self.part_dep_df.query(f"feature == '{self._list_to_key(fields)}'")
        dep_var = self.dep_var

        df_copy = df.copy()
        df_copy['feature'] = df_copy['feature'].str.slice(0, 45)
        df_copy = df_copy.sort_values(by=dep_var, ascending=asc)[:limit].sort_values(by=dep_var, ascending=not (asc))
        colors = prepare_colors(df_pd=df_copy)
        ax = df_copy.plot.barh(x="value", y=dep_var, sort_columns=True, figsize=(10, 10), color=colors)
        ax.set_ylabel(fields)
        for (p, t) in zip(ax.patches, df_copy['times']):
            ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y() * 1.005))
            ax.annotate(f'{int(t)}', ((p.get_width() * .45), p.get_y() + 0.1), color='white', weight='bold')
            
            
            
            
class EmbeddingsViz(Interpret):
    """
    Build the embeddings interpretation object
    """
    def __init__(self, learn, df):
        super().__init__(learn, df)
        self.cat_cols = self._get_cat_columns(is_wo_na=True)
        self.cl2idx = self._get_rev_emb_idxs()
        self.emds = self.learn.model.embeds
        self.classes_dict = self._get_categorify_obj()
        self.emb_map = self._get_embs_map()
        self.reduced_dim_df = None

    def _get_categorify_obj(self):
        """
        Scans learner processes and returning categorify object
        """
        for proc in self.learn.dls.train.procs:
            if (type(proc) == Categorify):
                return proc
        return None

    def _get_rev_emb_idxs(self):
        """
        Category to it's index dict.
        Is needed to transform cat class to it's digital representation (index) that is suitable for the model
        """
        classes = self._get_cat_columns()
        rev_dict = {}
        for i, c in enumerate(classes):
            if (c[-3:] != "_na"):
                rev_dict[c] = i
        return rev_dict

    def _get_emb_outp(self, field: str, field_class_idx: int):
        """
        Get embedding output for already encoded value of one field
        """
        emb = self.emds[self.cl2idx[field]]
        return emb(torch.tensor(field_class_idx, device=default_device())).data

    def _get_embs_map(self) -> OrderedDict:
        '''
        Output embedding vector for every item of every categorical column as a dictionary of dicts

        '''
        result = OrderedDict()

        for cat in self.cat_cols:
            cat_res = OrderedDict()
            for i, val in enumerate(self.classes_dict[cat]):
                cat_res[val] = self._get_emb_outp(field=cat, field_class_idx=i)
            result[cat] = cat_res

        return result

    def _add_time_col(self, source_df, class_col, feature_col=None, feature=None):
        '''
        Adds to embeddings map dataframe new column with times of value's number of occurrences
        Usefull for estimation of how accurate the value is (more time means more sure you can be)
        '''
        df = self.df.copy()
        source_df = source_df.copy()
        is_added = False
        if isNone(feature_col):
            feature_col = '#####addded####'
            source_df[feature_col] = feature
            is_added = True

        times = np.zeros(len(source_df))
        last_feat = ''
        vc = None
        for i, (f, v) in enumerate(zip(source_df[feature_col], source_df[class_col])):
            if (f != last_feat):
                vc = df[f].value_counts(dropna=False)
                vc.index = vc.index.map(str)
                last_feat = f
            if (v != '#na#'):
                times[i] = vc[str(v)]
            else:
                times[i] = vc['nan'] if ('nan' in vc.index) else 0
        source_df['times'] = times
        if (is_added == True):
            source_df.drop(columns=[feature_col], inplace=True)
        return source_df

    def _emb_map_reduce_dim(self, outp_dim: int = 2, method: str = 'pytorch', exclude: list = None):
        '''
        Reduces dimention of embedding map upto outp_dim
        Can use 'pytorch' approach (pca)
        or 'scilearn' for manifold.TSNE (longer)
        '''
        emb_map = self.emb_map
        exclude = listify(exclude)
        result = OrderedDict()
        for feat, val in emb_map.items():
            reformat = []
            names = []
            for k, v in val.items():
                reformat.append(v)
                names.append(k)
            reformat = torch.stack(reformat)
            if (exclude is not None) and (feat in exclude):
                continue
            if (method == 'scilearn'):
                tsne = manifold.TSNE(n_components=outp_dim, init='pca')
                reduced = tsne.fit_transform(to_np(reformat))
            else:
                reduced = reformat.pca(outp_dim)
            record = OrderedDict({k: v for k, v in zip(names, reduced)})
            result[feat] = record

        # to df
        data = []
        for feat, val in result.items():
            for k, v in val.items():
                dt = list(v) if (method == 'scilearn') else list(to_np(v))
                data.append([feat] + [k] + dt)
        names = ['feature', 'value'] + ['axis_' + str(i) for i in range(outp_dim)]
        result = pd.DataFrame(data, columns=names)

        self.reduced_dim_df = self._add_time_col(source_df=result, feature_col='feature', class_col='value')

    def _plot_2d_emb(self, feature: str, top_x: int = 10):
        """
        Plots feature embeddings in reduced dimension 
        """
        emb_map = self.reduced_dim_df
        sub_df = emb_map.query(f"feature == '{feature}'").sort_values('times', ascending=False).head(top_x)
        X = sub_df['axis_0']
        Y = sub_df['axis_1']
        plt.figure(figsize=(15, 8))
        plt.scatter(X, Y)
        for name, x, y in zip(sub_df['value'], X, Y):
            plt.text(x, y, name, color=np.random.rand(3) * 0.7, fontsize=11)
        plt.show()

    def plot_embedding(self, feature: str, top_x: int = 10, method: str = 'pytorch', exclude: list = None):
        """
        Plots the embedding in reduced (2) dimenstion for a particular `feature`
        `method` can use 'pytorch' approach (pca) or 'scilearn' for manifold.TSNE (longer)   
        `top_x` show only X top classes by number of occurences
        `exclude` exclude these values
        """
        df_map = ifNone(self.reduced_dim_df, self._emb_map_reduce_dim(method=method, exclude=exclude))
        self._plot_2d_emb(feature=feature, top_x=top_x)

    def measure_eucl_dists(self, feature: str, top_x=20):
        """
        Measures and returns Euclidean Distances between embeddings of a field
        Can use only `top_x` classes by number of occurences
        """
        emb_map = self.emb_map
        vectors = emb_map[feature]
        vectors = pd.DataFrame([[k, v] for k, v in vectors.items()], columns=['Class', 'Vector'])
        vectors = self._add_time_col(source_df=vectors, feature=feature, class_col='Class')
        vectors = vectors.sort_values(by='times', ascending=False).head(top_x)
        result = pd.DataFrame(columns=['Features', 'Euclidean Distance'])
        for _, r1 in vectors.iterrows():
            k1, v1 = r1['Class'], r1['Vector']
            for _, r2 in vectors.iterrows():
                k2, v2 = r2['Class'], r2['Vector']
                if (k1 != k2) and not (result['Features'].isin([f"{k2} vs {k1}"]).any()):
                    result = result.append(
                        {'Features': f"{k1} vs {k2}", 'Euclidean Distance': float(torch.dist(v1, v2))},
                        ignore_index=True)
        return result.sort_values(by='Euclidean Distance', ascending=True)