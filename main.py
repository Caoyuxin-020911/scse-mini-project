import time
from utils import DataAnalysis


def ask_column(cols):
    prompt = f"Please select a column (column name or integer in [0, {len(cols) - 1}]): "
    col_input = input(prompt)
    while not (col_input in cols or col_input.isdigit() and 0 <= int(col_input) < len(cols)):
        col_input = input(prompt)
    return col_input if col_input in cols else cols[int(col_input)]


def main():
    csv_filepath = input('File path: ')
    analysis = DataAnalysis(csv_filepath)
    print()
    analysis.show_informations()
    print()

    methods = ['DROP', 'MODE', 'MEDIAN', 'MEAN', 'BFILL', 'FFILL']
    prompt = f"Please select a method in {methods}: "
    while analysis.any_missing():
        analysis.show_informations()
        print()
        print('Please handle missing values.')
        col = ask_column(analysis.df.columns)
        choice = input(prompt)
        while not choice.upper() in methods:
            choice = input(prompt)
        analysis.handle_missing(col, choice)
        print(f"Missing values in {col} have been handled.")
        print()

    analysis.show_informations()
    print()
    analysis.handle_duplicates()
    print('Duplicates have been handled.')
    print()

    while True:
        try:
            analysis.show_informations()
            print()
            analysis.analyse_columns()
            print()
            print('0. Exit.')
            print('1. Plot variable distribution.')
            print('2. Check normality.')
            print('3. Conduct ANOVA.')
            print('4. Conduct Kruskal-Wallis.')
            print('5. Conduct T-Test.')
            print('6. Conduct Mann-Whitney U Test.')
            print('7. Conduct Chi-Square Test.')
            print('8. Conduct Regression.')
            print('9. Conduct sentiment analysis.')
            print()
            choice = input('Please choose (integer in [0, 9]): ')
            while not (choice.isdigit() and 0 <= int(choice) <= 9):
                choice = input('Please choose (integer in [0, 9]): ')
            choice = int(choice)
            if 0 == choice:
                break
            elif 1 == choice:
                col = ask_column(analysis.df.columns)
                if analysis.df_types1.at[col, 'type'] == "nominal":
                    col_count = analysis.df[col].value_counts().reset_index()
                    col_count.columns = ['category', 'count']
                    analysis.barplot(col_count, x='category', y='count')

                elif analysis.df_types1.at[col, 'type'] == "ordinal":
                    analysis.boxplot(x=col)

                elif analysis.df_types1.at[col, 'type'] == "interval_or_ratio":
                    analysis.histplot(x=col, kde=True)

                else:
                    print("There's an error!")

            elif 2 == choice:
                col = ask_column(analysis.df.columns)
                analysis.check_normality(col)
            elif 3 == choice:
                print('Column:')
                col = ask_column(analysis.df.columns)
                print('Group by:')
                groupby = ask_column(analysis.df.columns)
                analysis.anova(col, groupby)
            elif 4 == choice:
                print('Column:')
                col = ask_column(analysis.df.columns)
                print('Group by:')
                groupby = ask_column(analysis.df.columns)
                analysis.kruskal_wallis(col, groupby)
            elif 5 == choice:
                print('Column:')
                col = ask_column(analysis.df.columns)
                print('Group by:')
                groupby = ask_column(analysis.df.columns)
                analysis.t_test(col, groupby)
            elif 6 == choice:
                print('Column:')
                col = ask_column(analysis.df.columns)
                print('Group by:')
                groupby = ask_column(analysis.df.columns)
                analysis.mann_whitney_u_test(col, groupby)
            elif 7 == choice:
                print('Column:')
                col = ask_column(analysis.df.columns)
                print('Group by:')
                groupby = ask_column(analysis.df.columns)
                analysis.chi_square_test(col, groupby)
            elif 8 == choice:
                print('Column x:')
                col_x = ask_column(analysis.df.columns)
                print('Column y:')
                col_y = ask_column(analysis.df.columns)
                analysis.regression(col_x, col_y)
            elif 9 == choice:
                print('Text column:')
                col = ask_column(analysis.show_text_columns())
                print('0. Go back.')
                print('1. Vader.')
                print('2. Text-Blob.')
                print('3. Distilbert.')
                print()
                choice = input('Please choose (integer in [0, 3]):')
                while not (choice.isdigit() and 0 <= int(choice) <= 3):
                    choice = input('Please choose (integer in [0, 3]):')
                choice = int(choice)
                if 0 == choice:
                    pass
                elif 1 == choice:
                    analysis.vader(col)
                elif 2 == choice:
                    analysis.textblob(col)
                elif 3 == choice:
                    analysis.distilbert(col)
        except Exception as e:
            print(e)
        print()
        time.sleep(1)
    print('Goodbye.')
    print()


if __name__ == '__main__':
    main()
