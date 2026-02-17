import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import streamlit as st
from sklearn.model_selection import GridSearchCV
import io


@st.cache_data

def load_data():
    df=pd.read_csv("C:/Users/Harsh/PyCharmMiscProject/loan prediction/Loan_approval_data_2025.csv")
    return df
df = load_data()
st.header("First Tick prediction and enter data then tick model build for building and choosing model",text_alignment='justify')
c1 = st.sidebar.checkbox("Show data")
if c1:
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Original Data</h3>", unsafe_allow_html=True)
    st.dataframe(df)
c12 = st.sidebar.checkbox("EDA Report")
if c12:
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Top Five Rows</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(5))
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Last Five Rows</h3>",
                unsafe_allow_html=True)
    st.dataframe(df.tail(5))
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>All Column Info</h3>",
                unsafe_allow_html=True)
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>All Column Numeric Description</h3>",
                unsafe_allow_html=True)
    st.write(df.describe())
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Null Value Count</h3>",
                unsafe_allow_html=True)
    st.write(df.isnull().sum())

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Age Distribution</h3>",
                unsafe_allow_html=True)
    fig =  px.histogram(df,'age',nbins=8)
    fig.update_layout(
       xaxis_title="<b>Age</b>",
       yaxis_title="<b>count</b>",
    )
    fig.update_traces(
       marker_color="teal",
       marker_line_color="black",
       marker_line_width=1.5,
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Years Employeed Distribution</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, 'years_employed', nbins=8)
    fig.update_layout(
        xaxis_title="<b>Years</b>",
        yaxis_title="<b>count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Annual Income Distribution</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, 'annual_income', nbins=5)
    fig.update_layout(
        xaxis_title="<b>Annual Income</b>",
        yaxis_title="<b>count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Credit Score Distribution</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, 'credit_score', nbins=7)
    fig.update_layout(
        xaxis_title="<b>Credit Score</b>",
        yaxis_title="<b>count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Count of Default on File</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, 'defaults_on_file', nbins=2,histnorm='percent')
    fig.update_layout(
        xaxis_title="<b>Defaults on File</b>",
        yaxis_title="<b>count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Count of delinquencies_last_2yrs</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, 'delinquencies_last_2yrs', nbins=8, histnorm='percent')
    fig.update_layout(
        xaxis_title="<b>delinquencies_last_2yrs</b>",
        yaxis_title="<b>count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Count of derogatory_marks</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, 'derogatory_marks', nbins=5, histnorm='percent')
    fig.update_layout(
        xaxis_title="<b>derogatory_marks</b>",
        yaxis_title="<b>count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>count of loan intent categories</h3>",
                unsafe_allow_html=True)
    d = df['loan_intent'].value_counts().reset_index()
    fig = px.bar(d,x='loan_intent',y='count')
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    fig.update_layout(
        xaxis_title="<b>loan_intent</b>",
        yaxis_title="<b>count</b>",
    )
    st.plotly_chart(fig)

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>count of loan approval</h3>",
                unsafe_allow_html=True)
    d = df['loan_status'].value_counts().reset_index()
    fig = px.bar(d,x='loan_status',y='count')
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    fig.update_layout(
        xaxis_title="<b>loan_status</b>",
        yaxis_title="<b>count</b>",

    )
    st.plotly_chart(fig)


df1 = df.copy()
encoder = LabelEncoder()
df1['occupation_status'] = encoder.fit_transform(df1['occupation_status'])
df1['product_type'] = encoder.fit_transform(df1['product_type'])
df1['loan_intent'] = encoder.fit_transform(df1['loan_intent'])
scaler= StandardScaler()
x = df1.drop(['loan_status','customer_id','payment_to_income_ratio','loan_to_income_ratio','debt_to_income_ratio','savings_assets'],axis=1)
y = df1['loan_status']
x = scaler.fit_transform(x)
test=st.sidebar.slider('test size',0.0,1.0,0.0,0.1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

xg = XGBClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8]
 }

grid_search = GridSearchCV(
    estimator=xg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=2,
    n_jobs=1,
    refit=True,
 )

v3 = st.sidebar.checkbox("model build ")
if v3:
 modelname = st.sidebar.selectbox("select model",['linear','xgb','decisontree','randomforest','gradientboost','adaboost','svm','knc','gusianNB','gridsearchmodel'])
 if modelname == 'linear':
    linear = LogisticRegression()
    model = linear.fit(x_train, y_train)
 elif modelname == 'xgb':
    xgb = XGBClassifier()
    model = xgb.fit(x_train, y_train)
 elif modelname == 'decisontree':
    decissiontree = DecisionTreeClassifier()
    model =decissiontree.fit(x_train, y_train)
 elif modelname == 'randomforest':
    randomforest = RandomForestClassifier()
    model = randomforest.fit(x_train, y_train)
 elif modelname == 'gradientboost':
    gradientbooster = GradientBoostingClassifier()
    model=gradientbooster.fit(x_train, y_train)
 elif modelname == 'adaboost':
    adaboost = AdaBoostClassifier()
    model=adaboost.fit(x_train, y_train)
 elif modelname == 'svm':
    svm = SVC()
    model=svm.fit(x_train, y_train)
 elif modelname == 'knc':
    knc = KNeighborsClassifier()
    model=knc.fit(x_train, y_train)
 elif modelname == 'gusianNB':
    gusianNB = GaussianNB()
    model=gusianNB.fit(x_train, y_train)
 elif modelname == 'gridsearchmodel':
    model=grid_search.fit(x_train, y_train)


 def result(model):
    global ypred
    ypred = model.predict(x_test)
    acc =accuracy_score(y_test,ypred)
    return acc

 accuracy = result(model)
 b2 = st.sidebar.button("view result")
 if b2:
    accuracy = accuracy
    st.write(f"result is {accuracy}")



# Put this at the top of your Streamlit file (after imports)
occupation_map = {
    'Employed': 0,
    'Student': 2,
    'Self-Employed': 1
}

product_map = {
    'Credit Card': 0,
    'Personal Loan': 2.,
    'Line of Credit': 1
}

intent_map = {
    'Education': 2,
    'Debt Consolidation': 1,
    'Personal': 5,
    'Home Improvement': 3,
    'Business': 0,
    'Medical': 4
}

col1 , col2 = st.columns(2)
c2 = st.sidebar.checkbox("prediction")
if c2:
    age = col1.number_input("Age",1,80)
    occupation_status=col2.selectbox("occupation status",['Employed','Student','Self-Employed',''])
    years_employed=col1.number_input("Years employed",0.0,60.0)
    annual_income=col2.number_input("Annual Income",0,13000000)
    credit_score=col1.number_input("credit score",350,850)
    credit_history_years = col2.number_input('Credit History (Years)', min_value=0.0, max_value=40.0, value=5.0)
    current_debt = col1.number_input('Current Debt ', min_value=0.0, value=10000.0)
    defaults_on_file = col2.selectbox('Defaults on File', [0, 1])
    delinquencies_last_2yrs = col1.number_input('Delinquencies Last 2 Years', min_value=0, max_value=20, value=0)
    derogatory_marks = col2.number_input('Derogatory Marks', min_value=0, max_value=10, value=0)
    product_type = col1.selectbox('Product Type', ['Credit Card','Personal Loan','Line of Credit'])
    loan_intent = col2.selectbox('Loan Intent', ['Education','Debt Consolidation','Personal','Home Improvement','Business','Medical'])
    loan_amount = col1.number_input('Loan Amount Requested ', min_value=1000.0, value=10000000.0)
    interest_rate = col2.number_input('Interest Rate (%)', min_value=0.0, max_value=30.0, value=10.0)


    check = pd.DataFrame({

    'age' : [age],
    'occupation_status' : [occupation_map[occupation_status]],
    'years_employed' : [years_employed],
    'annual_income' : [annual_income],
    'credit_score' : [credit_score],
    'credit_history_years' : [credit_history_years],
    'current_debt' : [current_debt],
    'defaults_on_file' : [defaults_on_file],
    'delinquencies_last_2yrs' : [delinquencies_last_2yrs],
    'derogatory_marks' : [derogatory_marks],
    'product_type' : [product_map[product_type]],
    'loan_intent' : [intent_map[loan_intent]],
    'loan_amount' : [loan_amount],
    'interest_rate' : [interest_rate]
      })


    check1=scaler.transform(check)
    a1 = st.button("Ans")
    if not v3:
         st.error("please select model before ans button")
    else:
      if a1:
        Y_pre = model.predict(check1)
        if Y_pre[0]==1:
            st.success("congratulation, loan approved")
        elif Y_pre[0]==0:
            st.write("sorry, loan rejected")


