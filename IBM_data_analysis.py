import pandas as pd


def clean_dataset(df):
    #df = pd.read_csv("IBM_Employee_HR.csv")

    #Cleaning

    def encode(x):
        if x == "Yes":
            return 1
        else:
            return 0

    #Remove unnecessary spaces from columns names, if any
    df.columns = df.columns.str.strip()

    #Remove exact duplicates, if any
    df = df.drop_duplicates()

    #Remove empty column vales, if any
    numeric_columns = df.select_dtypes(include=["int64","float64"]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    #Remove unecessary columns (noisy data)
    df = df.drop(columns=["EmployeeNumber","EmployeeCount","StandardHours","Over18","Education","EducationField","StockOptionLevel"])

    #Apply encoding to 'Attrition' columns so that ML can work with it
    df["Attrition"] = df["Attrition"].apply(encode)


    #Get all the nominal (labeled) columns, and one-hot encode them
    #so that ML can work with them
    nominal_columns = ["BusinessTravel","Department","Gender","JobRole","MaritalStatus","OverTime"]
    df = pd.get_dummies(df,columns=nominal_columns)

    print(df.info())

    df.to_csv("IBM_Employee_HR_Cleaned.csv")
    return df

#clean_dataset(pd.read_csv("IBM_Employee_HR.csv"))
