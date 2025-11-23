import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import IBM_data_analysis
import split_data
import IBM_DATABASE
from PIL import Image

# Set Streamlit page layout to use the full browser width
st.set_page_config(layout="wide")

st.markdown("""
            <style>
                .stApp {
                    background-color:rgb(0,0,25);
                }

                div[data-testid='stToolbar'] {
                    background-color:white;
                }

                div[data-testid='stAppDeployButton'] button {
                    background-color:black;
                }

                span[data-testid='stMainMenu'] button {
                    background-color:black;
                }
                
            </style>
            """,unsafe_allow_html=True)

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

st.title("IBM Employee Attrition Model")

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs(["Upload dataset","Cleaned dataset","Summary statistics","Visualizations",
                                         "Training the ML Models","Evaluation","Insights","Prediction","Team"])


with tab1:
    file = st.file_uploader("Upload the CSV file",type="csv")

    clicked = st.button("Upload local dataset")

    if clicked:
        df = pd.read_csv("IBM_Employee_HR.csv")
        st.dataframe(df)

        if st.session_state.cleaned_df is None:
            st.session_state.cleaned_df = IBM_data_analysis.clean_dataset(df)

            if st.session_state.cleaned_df is not None:
                st.write("\n\nCleaned data successfully")
        

    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df)
        
        if st.session_state.cleaned_df is None:
            st.session_state.cleaned_df = IBM_data_analysis.clean_dataset(df)

            if st.session_state.cleaned_df is not None:
                st.write("\n\nCleaned data successfully")
    else:
        st.warning("Please upload a file")



with tab2:
    if st.session_state.cleaned_df is not None:
        st.dataframe(st.session_state.cleaned_df)


with tab3:
    if st.session_state.cleaned_df is not None:
        st.subheader("Summary of data : ")
        buffer = io.StringIO()
        st.session_state.cleaned_df.info(buf=buffer)
        st.code(buffer.getvalue())
        
        st.markdown("<br><br>",unsafe_allow_html=True)
        
        st.subheader("Summary statistics of numerical columns : ")
        st.write(st.session_state.cleaned_df.describe())
        


with tab4:
    if st.session_state.cleaned_df is not None:
        graphChoices =  ["Bar Chart","Heat Map","Histogram","Pie Chart","Boxplot","Bar Plot","Line Plot","Kernel Density Estimation Graph",
                         "Count Plot"]
        graph_choice = st.selectbox("Choose a graph",graphChoices)


        col1,col2 = st.columns(2)

        with col1:
            if graph_choice == "Bar Chart":
                plt_bar = IBM_DATABASE.bar_chart(st.session_state.cleaned_df)
                st.pyplot(plt_bar)

            elif graph_choice == "Heat Map":
                plt_heatmap = IBM_DATABASE.heatmap(st.session_state.cleaned_df)
                st.pyplot(plt_heatmap)
            
            elif graph_choice == "Histogram":
                plt_hist = IBM_DATABASE.histogram(st.session_state.cleaned_df)
                st.pyplot(plt_hist)

            elif graph_choice == "Pie Chart":
                plt_piechart = IBM_DATABASE.pie(st.session_state.cleaned_df)
                st.pyplot(plt_piechart)
                
            elif graph_choice == "Boxplot":
                plt_boxplot = IBM_DATABASE.boxplot(st.session_state.cleaned_df)
                st.pyplot(plt_boxplot)

            elif graph_choice == "Bar Plot":
                plt_barplot = IBM_DATABASE.barPlot(st.session_state.cleaned_df)
                st.pyplot(plt_barplot)

            elif graph_choice == "Line Plot":
                plt_lineplot = IBM_DATABASE.line_plot(st.session_state.cleaned_df)
                st.pyplot(plt_lineplot)

            elif graph_choice == "Kernel Density Estimation Graph":
                plt_kde = IBM_DATABASE.age_kde_plot(st.session_state.cleaned_df)
                st.pyplot(plt_kde)

            elif graph_choice == "Count Plot":
                plt_countplot = IBM_DATABASE.overtime_attrition_plot(st.session_state.cleaned_df)
                st.pyplot(plt_countplot)
            
            #code for  graph based on selection

        with col2:
            #explanation based on selection
            
            if graph_choice == "Bar Chart":
                st.subheader("Compares the average salary of employees who stayed vs those who left.")
                st.write("Employees who left tend to have slightly lower monthly income.\
                         This suggests salary may influence attrition, but it’s not the strongest factor.")

            elif graph_choice == "Histogram":
                st.subheader("How many employees fall into each age group.")


            elif graph_choice == "Boxplot":
                st.subheader("Compares the distribution of tenure between employees who stayed vs those who left.")
                st.write("People who leave usually have low tenure (few years at the company).\
                            This is a strong HR insight → new employees are more likely to leave.")

            elif graph_choice == "Heat Map":
                st.write("This heatmap shows the correlation between Attrition and all numerical features.\
                        \n\nRed cells indicate a positive relationship, blue cells indicate a negative relationship,\
                         and lighter colors show weaker correlations. \n\nThe values inside each box represent the exact\
                         correlation coefficient.")
            
            elif graph_choice == "Pie Chart":
                st.subheader("How many employees left vs stayed.")


            elif graph_choice == "Bar Plot":
                st.subheader("Which job roles experience the highest attrition percentage.")

            elif graph_choice == "Kernel Density Estimation Graph":
                st.subheader("Ages of people who left.")
                st.write("Younger employees (25–35) tend to leave more often than older employees.\
                            This fits real-world HR behavior — early-career employees job-hop more frequently.")


            elif graph_choice == "Line Plot":
                st.subheader("Attrition rate increases or decreases with commuting distance.")
                st.write("There is a clear pattern — employees with longer commutes tend to leave more often.")


            elif graph_choice == "Count Plot":
                st.subheader("How many employees leave depending on whether they work overtime.")
                st.write("This is the strongest factor.\
                            Employees who work overtime show significantly higher attrition.")

            



with tab5:
    col1,col2 = st.columns(2)

    if "logistic_model_results" not in st.session_state:
        st.session_state.logistic_model_results = None

    if "decisiontree_model_results" not in st.session_state:
        st.session_state.decisiontree_model_results = None

    with col1:
        clicked_logistic_regression = st.button("Train Logistic Regression model")
        if clicked_logistic_regression:

            if st.session_state.cleaned_df is not None:
                
                accuracy,conf_matrix,classReport = split_data.logistic_regression(0)
                st.session_state.logistic_model_results = (accuracy,conf_matrix,classReport)


        if st.session_state.logistic_model_results is not None:

            accuracy, conf_matrix, classReport = st.session_state.logistic_model_results
            st.write("Accuracy score : ",accuracy)
            st.write("Confusion Matrix\n",conf_matrix)
            st.markdown("<br>",unsafe_allow_html=True)
            st.write("\nClassification Report")
            st.table(classReport)
            
            st.markdown("<br><br>",unsafe_allow_html=True)

            st.write(f"The overall accuracy of the model is {accuracy*100:.2f}%")
            precision_stay = classReport.loc['0','precision']
            precision_leave = classReport.loc['1','precision']
            st.write(f"{precision_stay*100:.2f}% precise in identifying employees that stay")
            st.write(f"{precision_leave*100:.2f}% precise in identifying employees that leave")
            st.subheader(f"Confusion Matrix")
            st.write(f"Correctly identified {conf_matrix[0][0]} as staying (True Negative)")
            st.write(f"People staying but identifed as leaving : {conf_matrix[0][1]} (False Positive)")
            st.write(f"People leaving but identified as staying : {conf_matrix[1][0]} (False Negative)")
            st.write(f"Correctly identified {conf_matrix[1][1]} as leaving (True Positive)")

             
    with col2:
        clicked_decision_tree = st.button("Train Decision Tree model")
        if clicked_decision_tree:
            
            if st.session_state.cleaned_df is not None:
                accuracy,conf_matrix,classReport = split_data.decision_tree()
                st.session_state.decisiontree_model_results = (accuracy,conf_matrix,classReport)

        if st.session_state.decisiontree_model_results is not None:

            accuracy, conf_matrix, classReport = st.session_state.decisiontree_model_results
            st.write("Accuracy score : ",accuracy)
            st.write("Confusion Matrix\n",conf_matrix)
            st.markdown("<br>",unsafe_allow_html=True)
            st.write("\nClassification Report")
            st.table(classReport)

            st.markdown("<br><br>",unsafe_allow_html=True)
            
            st.write(f"The overall accuracy of the model is {accuracy*100:.2f}%")
            precision_stay = classReport.loc['0','precision']
            precision_leave = classReport.loc['1','precision']
            st.write(f"{precision_stay*100:.2f}% precise in identifying employees that stay")
            st.write(f"{precision_leave*100:.2f}% precise in identifying employees that leave")
            st.subheader(f"Confusion Matrix")
            st.write(f"Correctly identified {conf_matrix[0][0]} as staying (True Negative)")
            st.write(f"People staying but identifed as leaving : {conf_matrix[0][1]} (False Positive)")
            st.write(f"People leaving but identified as staying : {conf_matrix[1][0]} (False Negative)")
            st.write(f"Correctly identified {conf_matrix[1][1]} as leaving (True Positive)")
            

with tab6:
    if st.session_state.cleaned_df is not None:
        st.subheader("ROC Curve")
        clicked = st.button("Evaluate Logistic Regression Model")

        if clicked:

            col1,col2 = st.columns(2)

            with col1:
                fpr, tpr, auc_score = split_data.logistic_regression(1)

                    # --- ROC Curve ---

                fig, ax = plt.subplots(figsize=(3,3))
                ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
                ax.plot([0,1], [0,1], linestyle='--', color='gray')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()

                st.pyplot(fig,use_container_width=False)

            with col2:
                st.write("The ROC curve shows how well the model distinguishes between employees who stay and employees who leave.\
                        \n\nIt shows the trade-off between detecting employees who leave (TPR) and mistakenly predicting someone will leave\
                        when they won’t (FPR). A curve closer to the top-left means a better model.")
                st.markdown("---")

                st.write("AUC score Measures how well the model separates 'leave' vs 'stay'.")
                


with tab7:
    if st.session_state.cleaned_df is not None:
        st.subheader("Most important factors influencing attrition")
        important_factors = split_data.random_forest()

        top10 = important_factors.head(10)
        st.table(top10)

        st.markdown("<br><br>",unsafe_allow_html=True)

        st.write("The feature importance results show that OverTime, Monthly Income, Job Level, Work-Life Balance, and\
                 Age are the strongest predictors of employee attrition. \n\nEmployees who work overtime, earn lower salaries, or have poor\
                 work–life balance are significantly more likely to leave. Younger employees and those with lower job levels also show\
                 higher attrition rates. Overall, these factors contribute the most to whether an employee stays or resigns.")
    

with tab8:
    st.subheader("Employee Attrition Prediction")

    clicked_predict = st.button("Train and save Logistic Regression Model")

    if st.session_state.cleaned_df is not None:
        # Train the model (only needs to be done once)
        if clicked_predict:
            acc = split_data.train_and_save_logistic_model()
            st.success(f"Model trained and saved successfully! Accuracy = {acc*100:.2f}%")

        st.markdown("---")
        
        st.subheader("Enter Employee Details:")

        # USER INPUTS
        Age = st.slider("Age", 18, 60)
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000)
        YearsAtCompany = st.slider("Years at Company", 0, 40)
        OverTime = st.radio("OverTime", ["Yes", "No"])

        # Convert overtime to one-hot
        
        OverTime_Yes = 1 if OverTime == "Yes" else 0
        OverTime_No  = 1 - OverTime_Yes

        # Input list MUST follow model column order
        input_list = [Age, MonthlyIncome, YearsAtCompany, OverTime_No, OverTime_Yes]

        if st.button("Predict"):
            prediction = split_data.predict_attrition(input_list)

            if prediction == 1:
                st.error("Employee is LIKELY to leave the company.")
            else:
                st.success("Employee is NOT likely to leave.")

    

with tab9:

    st.subheader("Team Members")
    st.write("Abhishek Subramanian\t([LinkedIn](https://www.linkedin.com/in/abhishek-subramanian-64811b31a/))")
    st.write("Devinandana")
    st.write("Dhanush TD")
    st.write("Dony Binu")
    st.write("Vishnu Anup")
    st.write("Ibrahim P.K.")

    st.markdown("<br><br>",unsafe_allow_html=True)
    col1, col2, _ = st.columns([1,1,8])  # last column is just empty space

    with col1:
        image = Image.open("logo.png")
        st.image(image, width=80)

    with col2:
        image_two = Image.open("gt_logo.webp")
        st.image(image_two, width=80)

                
