import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline


class DataAnalysis(object):

    def __init__(self, csv_filepath):
        self.title = csv_filepath.split('\\')[-1].split('/')[-1]
        self.df = pd.read_csv(csv_filepath)
        self.df_types = None

    def show_informations(self):
        df = self.df
        print(self.title)
        combined_df = pd.concat([df.head(2), df.tail(2)])
        print(combined_df.to_string())
        print("Here is the Describle:")
        print(df.describe().to_string())
        print("Here is the Informations:")
        df.info()
        self.df = df

    def any_missing(self):
        df = self.df
        any = 0 < df.isna().sum().sum()
        self.df = df
        return any

    def handle_missing(self, column, method):
        df = self.df
        method = method.upper()
        if 'DROP' == method:
            df = df.dropna(subset=column, ignore_index=True)
        elif 'MODE' == method:
            df[column] = df[column].fillna(df[column].mode()[0])
        elif 'MEDIAN' == method:
            df[column] = df[column].fillna(df[column].median())
        elif 'MEAN' == method:
            df[column] = df[column].fillna(df[column].mean())
        elif 'BFILL' == method:
            df[column] = df[column].fillna(method='bfill')
        elif 'FFILL' == method:
            df[column] = df[column].fillna(method='ffill')
        self.df = df

    def handle_duplicates(self):
        df = self.df
        df = df.drop_duplicates(ignore_index=True)
        self.df = df

    def analyse_columns(self):
        df = self.df
        if self.df_types is None:
            n = len(df.columns)
            type_ls = [pd.NA] * n
            mode_ls = [pd.NA] * n
            median_ls = [pd.NA] * n
            mean_ls = [pd.NA] * n
            var_ls = [pd.NA] * n
            std_ls = [pd.NA] * n
            kurt_ls = [pd.NA] * n
            skew_ls = [pd.NA] * n
            for i, column in enumerate(df.columns):
                df_col = df[column]
                if pd.api.types.is_bool_dtype(df_col.dtype) or pd.api.types.is_object_dtype(df_col.dtype):
                    type_ls[i] = 'nominal'
                    mode_ls[i] = df_col.mode()[0]
                elif pd.api.types.is_datetime64_any_dtype(df_col.dtype) or df_col.nunique() / len(df_col) < 0.05:
                    type_ls[i] = 'ordinal'
                    mode_ls[i] = df_col.mode()[0]
                    median_ls[i] = df_col.median()
                else:
                    type_ls[i] = 'interval_or_ratio'
                    median_ls[i] = df_col.median()
                    mean_ls[i] = df_col.mean()
                    var_ls[i] = df_col.var()
                    std_ls[i] = df_col.std()
                    kurt_ls[i] = df_col.kurt()
                    skew_ls[i] = df_col.skew()
            self.df_types1 = pd.DataFrame({
                'name': df.columns.values,
                'dtype': df.dtypes.values,
                'type': type_ls,
                'mode': mode_ls,
                'median': median_ls,
                'mean': mean_ls,
            }, index=df.columns)

            self.df_types2 = pd.DataFrame({
                'name': df.columns.values,
                'dtype': df.dtypes.values,
                'type': type_ls,
                'var': var_ls,
                'std': std_ls,
                'kurt': kurt_ls,
                'skew': skew_ls,
            }, index=df.columns)
        print("Here's the analyse column:")
        print(self.df_types1.to_string())
        print("\n")
        print(self.df_types2.to_string())
        self.df = df

    def barplot(self, data, x ,y):
        df = self.df
        plt.figure()
        sns.barplot(data, x=x, y=y)
        plt.show()
        self.df = df

    def boxplot(self, x):
        df = self.df
        plt.figure()
        sns.boxplot(df, x=x)
        plt.show()
        self.df = df

    def histplot(self, x, kde):
        df = self.df
        plt.figure()
        sns.histplot(df, x=x, kde=kde)
        plt.show()
        self.df = df

    def scatterplot(self, x, y):
        df = self.df
        plt.figure()
        sns.scatterplot(df, x=x, y=y)
        plt.show()
        self.df = df

    def check_normality(self, column):
        df = self.df
        df_col = df[column]
        if len(df_col) <= 2000:
            print("Shapiro-Wilk Normality Test:")
            stat, p_value = stats.shapiro(df_col)
            print('Statistic:', stat)
            print('P-value:', p_value)
            if p_value < 0.05:
                print(f"'{column}' is not normally distributed (at the {5}% significance level).")
            else:
                print(f"'{column}' is normally distributed (at the {5}% significance level).")
        else:
            print("Anderson-Darling Normality Test:")
            anderson_result = stats.anderson(df_col, dist='norm')
            print('Statistic:', anderson_result.statistic)
            for sig_level, crit_value in zip(anderson_result.significance_level, anderson_result.critical_values):
                print(f"Significance level {sig_level}%: {crit_value}")
            if anderson_result.critical_values[2] < anderson_result.statistic:
                print(f"'{column}' is not normally distributed (at the {5}% significance level).")
            else:
                print(f"'{column}' is normally distributed (at the {5}% significance level).")
        sm.qqplot(df_col, line='s')
        plt.show()
        self.df = df

    def anova(self, column, groupby):
        df = self.df
        print("ANOVA:")
        stat, p_value = stats.f_oneway(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df = df

    def kruskal_wallis(self, column, groupby):
        df = self.df
        print("Kruskal-Wallis:")
        stat, p_value = stats.kruskal(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df = df

    def t_test(self, column, groupby):
        df = self.df
        print("T-Test:")
        stat, p_value = stats.ttest_ind(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df = df

    def mann_whitney_u_test(self, column, groupby):
        df = self.df
        print("Mann-Whitney U Test:")
        stat, p_value = stats.mannwhitneyu(*(df[df[groupby] == val][column] for val in df[groupby].unique()))
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df = df

    def chi_square_test(self, column, groupby):
        df = self.df
        print("Chi-Square Test:")
        contingency_table = pd.crosstab(df[column], df[groupby])
        stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"stat: {stat}")
        print(f"p-value: {p_value}")
        print(f"dof: {dof}")
        print(f"expected: {expected}")
        if p_value < 0.05:
            print('Reject the null hypothesis:')
        else:
            print('Fail to reject the null hypothesis:')
        print(f"\tThere is no difference in the average '{column}' across all kinds of '{groupby}'.")
        self.df = df

    def regression(self, column_x, column_y):
        df = self.df
        print("Regression:")
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[column_x], df[column_y])
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value ** 2}")
        print(f"P-value: {p_value}")
        print(f"Standard error: {std_err}")
        plt.figure()
        sns.scatterplot(df, x=column_x, y=column_y)
        plt.plot(df[column_x], intercept + slope * df[column_x], 'r', label='Fitted line')
        plt.show()
        self.df = df

    def show_text_columns(self):
        df = self.df
        columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
        print("Text Columns:")
        print(pd.DataFrame(
            {
                'Column Name': columns,
                'Average Entry Length': [df[col].apply(len).mean() for col in columns],
                'Unique Entries': [df[col].nunique() for col in columns],
            }
        ))
        self.df= df
        return columns

    def vader(self, column):
        df = self.df
        vader_analyzer = SentimentIntensityAnalyzer()
        res = df[column].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
        print("Vader Sentiment Analysis:")
        print(
            res,
            res.apply(lambda x: 'positive' if x >= +0.05 else 'negative' if x <= -0.05 else 'neutral'),
            sep='\n',
        )
        self.df = df

    def textblob(self, column):
        df = self.df
        res = df[column].apply(lambda x: TextBlob(x).sentiment)
        print("Text-Blob Sentiment Analysis:")
        print(
            res.apply(lambda x: x.polarity),
            res.apply(lambda x: 'positive' if x.polarity > 0 else 'negative' if x.polarity < 0 else 'neutral'),
            res.apply(lambda x: x.subjectivity),
            sep='\n',
        )
        self.df = df

    def distilbert(self, column):
        df = self.df
        sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        res = df[column].apply(lambda x: sentiment_pipeline(x)[0])
        print("Distilbert Sentiment Analysis:")
        print(
            res.apply(lambda x: x['score']),
            res.apply(lambda x: int(x['label'].split(' ')[0])).apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral'),
            sep='\n',
        )
        self.df = df
