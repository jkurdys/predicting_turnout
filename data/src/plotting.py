import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)

def plot_kde(df):

    fig, ax = plt.subplots(figsize = (12,8))

    sns.kdeplot(df.loc[(df['voted']==1), 'age'], color =  '#F8766D', shade = True, label= 'Voted')
    sns.kdeplot(df.loc[(df['voted']==0), 'age'], color =  '#619CFF', shade = True, label= 'Did Not Vote')

    plt.xlabel('Voter Age')
    plt.ylabel('Probability Density')

    ax.set(xlim=(8, 130))
    ax.legend(loc='upper right')
    ax.set_title("Turnout by Age", y=0.99, fontsize=24)

    plt.tight_layout(rect=(0,0,1,0.98))
    plt.savefig("../images/original_kde_plots.png", transparent=True, dpi=200)

def bar_categorical_plot(cat_df, cat_col, ax):
        """ Plots a 100% Fill Stacked Barchart of Churned vs Active users using the values passed.
            Args:
                cat_df (Pandas Dataframe): The predictors and target that you want to use for the 
                                        X axis on the barchart
                cat_col (str): The column name of the categorical data to plot
                ax (matplotlib axis): An axis to plot the barchat
            Returns:
                None
                Modifies ax (matplotlib axis): An axis with the barchart plot
        """
        group_data = cat_df.groupby([cat_col, 'voted']).size().unstack()
        group_data.columns = ['Non Voter', 'Voter']
        group_data = group_data.fillna(value=0)
        group_data['Eligible Voters'] = group_data['Non Voter'] + group_data['Voter']
        group_data['Did Not Vote'] = group_data['Non Voter'] / group_data['Eligible Voters']
        group_data['Voted'] = group_data['Voter'] / group_data['Eligible Voters']
        total_users = group_data.sort_values('Did Not Vote', ascending=False)['Eligible Voters']
        turnout = group_data[['Did Not Vote', 'Voted']].sort_values('Did Not Vote', ascending=False)
        turnout.plot.bar(stacked=True, ax=ax, color=['#619CFF', '#F8766D'], alpha = 0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        ax.set_xlabel('')
        ax.set_title(cat_col.replace('_', ' ').title())
        ax.set_ylabel('% of Eligible Voters')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(loc='upper right')

def plot_categorical_features(df):
        """ Visualize categorical predictors using stacked 100% fill barcharts
            Args: 
                None 
            
            Returns:
                None:
                Saves the barchart plots from the categorical features into a .png file for viewing
        """
        categorical_data = df
        categorical_columns = cat_cols
        
        fig, axs = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(20,35))

        for col, ax in zip(categorical_columns, axs.flatten()):
            bar_categorical_plot(categorical_data, col, ax)

        plt.tight_layout(rect=(0,0,1,0.98))
        plt.suptitle(f"100% Fill Barchart for Categorical Predictors", y=0.99, fontsize=35)
        plt.savefig("../images/original_barchart_plots.png", transparent=True, dpi=200)

def plot_one_categorical_feature(df, col):
        """ Visualize categorical predictors using stacked 100% fill barcharts
            Args: 
                None 
            
            Returns:
                None:
                Saves the barchart plots from the categorical features into a .png file for viewing
        """
        categorical_data = df
        cols = ['race', 'gender', 'congressional_district', 'county']
        
        fig, ax = plt.subplots(figsize = (12,6))
        
        bar_categorical_plot(categorical_data, cols[col], ax)

        plt.tight_layout(rect=(0,0,1,0.98))
        plt.savefig(f"../images/{cols[col]}_bar.png", transparent=True, dpi=200)


if __name__=='__main__':
    pass