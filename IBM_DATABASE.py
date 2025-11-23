import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ====================================================
# PART 2: VISUALIZATION (Matplotlib Only) 🎨
# ====================================================

def bar_chart(df):
    # Create human-readable label for visualizations (placed AFTER X and y definition)
    df['Attrition_Label'] = df['Attrition'].map({0: 'No Attrition', 1: 'Attrition'})


    # --- Existing Plots (1-5) ---

    # 1. Bar Graph: Average Monthly Income by Attrition Outcome
    fig ,ax = plt.subplots(figsize=(3,3))
    avg_income = df.groupby('Attrition_Label')['MonthlyIncome'].mean()
    
    ax.bar(avg_income.index, avg_income.values, color=['#1f77b4', '#d62728']) # Blue (No Attrition), Red (Attrition)
    ax.set_title('Average Monthly Income by Attrition Status')
    ax.set_xlabel('Attrition Status')
    ax.set_ylabel('Average Monthly Income ($)')
    return fig


def histogram(df):
    # 3. Histogram: Frequency of Age
    fig, ax = plt.subplots(figsize=(3,3))

    ax.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Histogram of Employee Ages')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')

    return fig


def boxplot(df):
    # 4. Box Plot: Years at Company by Attrition
    fig, ax = plt.subplots(figsize=(3,3))

    data_to_plot = [df[df['Attrition_Label'] == 'No Attrition']['YearsAtCompany'],
                    df[df['Attrition_Label'] == 'Attrition']['YearsAtCompany']]
    
    # Use distinct colors for boxes
    box_colors = ['#1f77b4', '#d62728']
    bp = ax.boxplot(data_to_plot, labels=['No Attrition', 'Attrition'], patch_artist=True,
                medianprops=dict(color='white')) # White median line for contrast

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title('Box Plot: Years At Company by Attrition Status')
    ax.set_xlabel('Attrition Status')
    ax.set_ylabel('Years at Company')
    return fig


def pie(df):
    # 5. Pie Plot: Proportion of Employees who Left (Attrition)
    fig, ax = plt.subplots(figsize=(3,3))
    
    outcome_counts = df['Attrition_Label'].value_counts()
    ax.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'])
    ax.set_title('Proportion of Employee Attrition')
    return fig


# --- New Plots (6-10) ---

def barPlot(df):
    # 6. Bar Plot (New): Attrition Rate by Job Role
    # Calculate the attrition rate for each Job Role (extracted from the OHE columns)
    job_role_cols = [col for col in df.columns if col.startswith('JobRole_')]
    job_roles = df[job_role_cols].sum(axis=0)
    attrition_by_role = df[df['Attrition'] == 1][job_role_cols].sum(axis=0)

    # Calculate rate: Attrition Count / Total Employees in Role
    attrition_rate = (attrition_by_role / job_roles)   #.sort_values(ascending=False)
    role_labels = [col.replace('JobRole_', '') for col in attrition_rate.index]

    # FIX: Use plt.colormaps instead of deprecated plt.cm.get_cmap
    cmap = plt.colormaps.get_cmap('viridis') 
    colors = [cmap(i / len(role_labels)) for i in range(len(role_labels))]

    fig, ax = plt.subplots(figsize=(3,3))
    
    ax.bar(role_labels, attrition_rate.values, color=colors)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Attrition Rate by Job Role')
    ax.set_xlabel('Job Role')
    ax.set_ylabel('Attrition Rate')
    #fig.tight_layout()
    return fig


def age_kde_plot(df):
    # Ensure proper labels
    #df['Attrition_Label'] = df['Attrition'].map({0: 'Stayed', 1: 'Left'})

    df_left = df[df["Attrition"] == 1]
    
    fig, ax = plt.subplots(figsize=(4,3))
    
    sns.kdeplot(
        data=df_left,
        x="Age",
        hue="Attrition_Label",
        fill=True,
        alpha=0.5
    )
    
    ax.set_title("Age Distribution of Employess Who Left")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    
    return fig


def overtime_attrition_plot(df):

    def reconstruct(x):
        if x == 1:
            return "Yes"
        else:
            return "No"

    df_temp = df.copy()
    df_temp["OverTime"] = df_temp["OverTime_Yes"].apply(reconstruct)
    
    plt.figure(figsize=(6,4))
    sns.countplot(data=df_temp, x='OverTime', hue='Attrition')
    plt.title("OverTime vs Attrition")
    plt.xlabel("OverTime")
    plt.ylabel("Number of Employees")
    plt.legend(title="Attrition", labels=["No", "Yes"])
    return plt


def line_plot(df):
    # 10. Line Plot (New): Distance From Home vs. Attrition Rate
    # Calculate the attrition rate for each distance
    distance_attrition = df.groupby('DistanceFromHome')['Attrition'].agg(['mean', 'count'])
    distance_attrition.rename(columns={'mean': 'Attrition Rate', 'count': 'Employee Count'}, inplace=True)

    # Smooth the line plot by taking a rolling mean (optional, but makes it clearer)
    rolling_rate = distance_attrition['Attrition Rate'].rolling(window=3, center=True).mean()

    fig, ax = plt.subplots(figsize=(5,5))
    
    ax.plot(rolling_rate.index, rolling_rate.values, color='purple', linewidth=3, marker='o', markersize=5)
    ax.set_title('Attrition Rate vs. Distance From Home (3-point Rolling Average)')
    ax.set_xlabel('Distance From Home (Miles)')
    ax.set_ylabel('Attrition Rate')
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig


def heatmap(df):
    # Select features including the target 'Attrition'
    numerical_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    numerical_cols.remove("Attrition")
    
    heatmap_features = ['Attrition'] + numerical_cols
    corr_matrix = df[heatmap_features].corr()

    plt.figure(figsize=(10, 8))
    # Use Matplotlib's imshow function for the heatmap, leveraging a color map
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

    # Set ticks and labels
    plt.xticks(ticks=np.arange(len(heatmap_features)), labels=heatmap_features, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(heatmap_features)), labels=heatmap_features)
    plt.title('Heatmap: Correlation Matrix of Key Features (vs. Attrition)')

    # IMPROVEMENT: Add correlation values to the cells with color contrast
    for i in range(len(heatmap_features)):
        for j in range(len(heatmap_features)):
            corr_val = corr_matrix.iloc[i, j]
            # Determine text color based on cell color (light vs dark)
            text_color = 'white' if abs(corr_val) > 0.6 else 'black'
            plt.text(j, i, f'{corr_val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()
    return plt


