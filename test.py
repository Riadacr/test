from abc import ABC
from cProfile import label
#from PIL.Image import alpha_composite
from altair.vegalite.v4.schema.channels import Key
import numpy as np
from numpy.core.fromnumeric import std
from numpy.lib.function_base import average
import pandas as pd
import datetime
from datetime import timedelta, date
import matplotlib.pyplot as plt
from pytz import HOUR
from sklearn.linear_model import LinearRegression
from timeit import default_timer as timer
import streamlit as st
from streamlit.proto.RootContainer_pb2 import SIDEBAR
from scipy.stats import norm, weibull_min
from scipy.ndimage import uniform_filter1d
from itertools import islice
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly_express as px
import plotly.figure_factory as ff
from plotly.colors import n_colors
from scipy.special import gamma, factorial
import io
from math import nan, isnan, floor, ceil
import base64
from enum import unique
import calendar
from statistics import median

st.set_option('deprecation.showPyplotGlobalUse', False)
options = ["Weibull Production Analysis","Ridgeline Diagram"]

def fileSelector():
    dSource = st.sidebar.file_uploader("Upload Data for Analysis",accept_multiple_files=False)
    return dSource

def sheetPicker(dSource):
    sheetNames = pd.ExcelFile(dSource)
    sheetNames = sheetNames.sheet_names
    st.sidebar.caption("Select the sheet containing the containing the pertinent data.")
    dataSheet = st.sidebar.selectbox("Available Sheets",options=sheetNames)
    return dataSheet

def columnPicker(dSource,dataSheet):
    columnNames = list(pd.ExcelFile(dSource).parse(dataSheet))
    columnNames = [x for x in columnNames if x != "nan"] #This code drops columns named nan.
    columnNames = [y for y in columnNames if "Unnamed" not in y] #This code drops columns that have data but are not named.
    selectedColumns = st.sidebar.multiselect("Select the Columns for Analysis and the Date Column",columnNames)
    st.sidebar.caption("""
        The user must select at least 2 columns: 

        1. A column containing date information.
        2. A column containing process data.
        """)
    return selectedColumns

def dateColumnPicker(selectedColumns):
    dateColumn = st.sidebar.selectbox("Select the Date column",selectedColumns)
    return dateColumn

def sorcery():
    
    dSource = fileSelector()

    if not dSource and 'dSource' in st.session_state:
        dSource = st.session_state['dSource']
    
    if dSource:
        st.session_state['dSource'] = dSource
        dataSheet = sheetPicker(dSource)
        
        if not dataSheet and 'dataSheet' in st.session_state:
            dataSheet = st.session_state['dataSheet']
        
        if dataSheet:
            st.session_state['dataSheet'] = dataSheet
            selectedColumns = columnPicker(dSource,dataSheet)
            
            if not selectedColumns and 'selectedColumns' in st.session_state:
                selectedColumns = st.session_state['selectedColumns']
            
            if selectedColumns:
                st.session_state['selectedColumns'] = selectedColumns
                dateColumn = dateColumnPicker(selectedColumns)

                if not dateColumn and 'dateColumn' in st.session_state:
                    dateColumn = st.session_state['dateColumn']
                if dateColumn:
                    source = {
                        'dSource':dSource,
                        'dataSheet':dataSheet,
                        'selectedColumns':selectedColumns,
                        'dateColumn':dateColumn
                    }
                    with st.sidebar.form("Uploader"):
                        if st.form_submit_button("Upload Data"):
                            return source

@st.experimental_memo
def DataCollector (dSource,dataSheet,selectedColumns,dateColumn):
    df = pd.read_excel(dSource,sheet_name=dataSheet)
    df =df[df.columns.intersection(selectedColumns)]
    df.set_index(dateColumn,inplace=True)
    return df

def AnalysisPicker ():
    analysis = st.radio(label="Analysis Menu", options=options,horizontal=True)
    return analysis

def offsetSelector():
    choices = [90,60,30]
    offset = st.selectbox("Select the minimum number of datapoints for analysis",options=choices)
    return offset

def sDateSelector(minDate,maxDate,offset):
    startDate = st.date_input(label="Select the start date for the analysis",value=minDate,min_value=minDate,max_value=maxDate-datetime.timedelta(days=offset))
    return startDate

def round_to_1(x):
    decimals = -int(floor(int(np.log10(abs(x)))))
    multiplier = 10 ** decimals
    return ceil(x * multiplier)/multiplier

def minValueSelector(minVal,maxVal,lower):
    if maxVal >= 100:
        if minVal >= 1.0:
            minVal = int(round_to_1(minVal))
        else:
            minVal = int(minVal)
        maxVal = int(round_to_1(maxVal))
        step = int(10)
    else:
        minVal = int(minVal)
        maxVal = int(maxVal)
        step = int((maxVal-minVal)/25)
    if -int(floor(int(np.log10(abs(lower))))) < 0:
        roundto = -1
    else:
        roundto = -int(floor(int(np.log10(abs(lower))))) + 1
    lower = round(lower, ndigits=roundto)
    minValue = st.slider("Select the lower bound for data filtering.", min_value=minVal,max_value=maxVal,value=int(lower),step=step)
    return minValue

def maxValueSelector(minVal,maxVal,upper):
    if maxVal >= 100:
        step = int(10)
        minVal = int(round_to_1(minVal))
        maxVal = int(round_to_1(maxVal))
    else:
        minVal = int(minVal)
        maxVal = int(maxVal)
        step = int((maxVal-minVal)/25)
    if -int(floor(int(np.log10(abs(upper))))) < 0:
        roundto = -1
    else:
        roundto = -int(floor(int(np.log10(abs(upper))))) + 1
    upper = round(upper, ndigits=roundto)
    maxValue = st.slider("Select the upper bound for data filtering.", min_value=minVal,max_value=maxVal,value=int(upper),step=step)
    return maxValue

def eDateSelector(maxDate,startDate,offset):
    endDate = st.date_input(label="Select the end date for the analysis",value=maxDate,min_value=startDate+datetime.timedelta(days=offset),max_value=maxDate)
    return endDate

def WeibullAnalysisControls(df):
    with st.form('weibull_controls'):
        # Set the minimum number of analysis days.  This constrains the selectable dates for analysis.
        offset = offsetSelector()
        # This section of code drops duplicate dates and replaces blanks with NA.
        df.drop_duplicates()
        na_value = float("NaN")
        # Determine the earliest and latest dates provided by the user.
        # These will define the date range available, and with the offset
        # they will be used to constrain the dates that are available for selection.
        df.replace("",na_value,inplace=True)
        minDate = min(df.index)
        maxDate = max(df.index)
        startDate = sDateSelector(minDate,maxDate,offset)
        endDate = eDateSelector(maxDate,startDate,offset)
        sDate = minDate.strftime("%m/%d/%Y")
        eDate = maxDate.strftime("%m/%d/%Y")
        analysisColumn = st.selectbox("Select the data to analyze.", options=df.columns)
        inputsWeibull = {
            'startDate':startDate,
            'endDate':endDate,
            'sDate':sDate,
            'eDate':eDate,
            'analysisColumn': analysisColumn
        }
        if st.form_submit_button('Run Weibull Analysis'):
            return inputsWeibull


def RidgelineInputs(df):
    with st.form('Ridgeline Diagram Controls'):
        # This section of code drops duplicate dates and replaces blanks with NA.
        df.drop_duplicates()
        na_value = float("NaN")
        df.replace("",na_value,inplace=True)
        analysisColumn_ridgeline = st.selectbox("Select the data to analyze.", options=df.columns)
        RidgelineAnalysisColumn = {
            'RidgelineAnalysisColumn':analysisColumn_ridgeline
        }
        if st.form_submit_button('Load the data for the Ridgeline Diagram'):
            return RidgelineAnalysisColumn

@st.experimental_memo
def WeibullAnalysisDPLCalculations(inputsWeibull,df):
    df1 = df[inputsWeibull['startDate']:inputsWeibull['endDate']]
    df1.replace(np.nan,0,inplace=True)
    data = df1
    data['Data'] = data[inputsWeibull['analysisColumn']]
    offset = int(len(data))
    #This section will sort the dataframe from low to high.
    WeibullData = data.sort_values(by=['Data'],ascending=False)
    #Assign a rank to the data.
    WeibullData['Number']=np.arange(len(WeibullData))
    WeibullData['Number'] = WeibullData['Number']+1
    #Determine the median rank for each datapoint.
    WeibullData['MedianRank'] = (len(WeibullData['Number']) - WeibullData['Number']+1-0.3)/(len(WeibullData['Number']*0.4))
    #Determine the transformed median rank for the Weibull plot.
    WeibullData['Plot_MedianRank'] = np.log(1/(1-WeibullData['MedianRank']))
    #Transform the production data.
    X_Axis = np.array(WeibullData['Data'])
    X_Axis=np.where(X_Axis==0,0.0001,X_Axis)
    WeibullData['Plot_XVals'] = np.log(X_Axis)

    #Determine the best regression.

    MinPoints = int(offset*0.1)                            #Min number of points to include in regression.
    Correl_Coef = 0                                        #Initialize storage variables.
    Best_Correl_Coef = 0                                  
    nPoints = 0
    MaxPoints = int(len(WeibullData['Plot_XVals']))        # Calculate the Maximum Points for Regression.

    # Dictionaries to store regression information.
    results = {
                'Slope':1,
                'Intercept':1
                }
    BestResults = {
                'Slope':1,
                'Intercept':1
                }           
    i = 0
    #Loop to determine the best regression.
    for iFirstPoint in range(MaxPoints):
        i = i + 1
        #Determine the size of the dataset.
        iBreakPoint = iFirstPoint + MinPoints 
        if iBreakPoint <= MaxPoints:
            for iBreakPoint in list(range(iFirstPoint,MaxPoints,1)):
                #Load the data points w/ Y transformation
                XPts = WeibullData['Plot_XVals'][iFirstPoint:iBreakPoint]
                YPts = np.log(WeibullData['Plot_MedianRank'][iFirstPoint:iBreakPoint])
                #print(len(range(iFirstPoint,iBreakPoint,1)))
                # Polynomial Regression
                if len(XPts)>=MinPoints:
                    x = np.array(XPts).reshape((-1,1))
                    y = np.array(YPts)
                    model = LinearRegression(fit_intercept=True).fit(x,y)
                    Correl_Coef = model.score(x,y)
                    results['Slope'] = model.coef_
                    results['Intercept'] = model.intercept_
                    if Correl_Coef > Best_Correl_Coef:
                        Best_Correl_Coef = Correl_Coef
                        BestResults['Slope'] = results['Slope']
                        BestResults['Intercept'] = results['Intercept']
                        nPoints = len(range(iFirstPoint,iBreakPoint))
                        #print(Best_Correl_Coef, iFirstPoint,iBreakPoint,nPoints)
    Eta = np.exp(-BestResults['Intercept']/BestResults['Slope'])
    Beta = float(BestResults['Slope'])

    # Create the Demonstrated Production Line
    # Calculate the Y values for the dpl.
    #dpl = pd.DataFrame(np.log(1/(1-levels)))
    dpl = pd.DataFrame(WeibullData['Plot_MedianRank'])
    dpl.columns = ['Y']
    # Calculate the constant for the linearization of the Weibull CDF.
    LinConst = Beta * np.log(Eta)
    # Calculate the X values for the dpl.
    dpl['X'] = np.exp((np.log(dpl['Y']) + LinConst) / Beta)
    WeibullDPL = {
        'WeibullData': WeibullData,
        'dpl': dpl,
        'Eta': Eta,
        'Beta': Beta,
        'data': data,
        'Best_Correl_Coef': Best_Correl_Coef,
        'nPoints': nPoints
    }
    return WeibullDPL

@st.experimental_memo
def WeibullRVLCalculations(WeibullDPL,rvl_beta):
    # Create the Reduced Variability Line
    rvl = pd.DataFrame(WeibullDPL['dpl']['Y'])
    rvl.columns = ['Y']
    rvl_beta = rvl_beta
    rvl_Const = rvl_beta * np.log(WeibullDPL['Eta'])
    rvl['X'] = np.exp((np.log(rvl['Y']) + rvl_Const) / rvl_beta)
    WeibullRVL = {
        'rvl':rvl
    }
    return WeibullRVL

def RVL_sorcery(WeibullDPL):
    with st.form("rvl_solver"):
        rvl_beta = st.slider('Select the Beta for the Reduced Variability Line',min_value=int(WeibullDPL['Beta']),max_value=50,value=25,step=1)
        if st.form_submit_button("Compute the Reduced Variability Line"):
            return rvl_beta

def NMPLT_sorcery(rvl_beta):
    with st.form("nmplt_solver"):
        nmplt_beta = st.slider('Select the Beta for the Name Plate Capacity Line',min_value=rvl_beta,max_value=50,value=50,step=1)
        if st.form_submit_button("Compute the Name Plate Capacity Line"):
            return nmplt_beta

@st.experimental_memo
def WeibullNMPLTCalculations(WeibullDPL,nmplt_beta):
    # Create the Nameplate line.
    # identify the maximum production value in the raw data.
    weibulldata_column = WeibullDPL['WeibullData']['Data']
    max_weibull_point = float(weibulldata_column.max())
    # identify the maximum production value of the DPL.
    dpldata_column = WeibullDPL['dpl']['X']
    max_dpl_point = dpldata_column.max()
    # choose the calibration point for the nameplate line.
    nameplate_calibration_point = min(max_dpl_point,max_weibull_point)
    # calculate the scale factor (eta) of the nameplate line.
    nameplate_beta = 50 #Accepts the user input for the shape factor to be used.
    weibull_plot_medianrank_column = WeibullDPL['WeibullData']['Plot_MedianRank'] #identify the formatted maximum median rank from the data set.
    max_weibull_plot_medianrank = float(weibull_plot_medianrank_column.max())
    A = nameplate_beta * np.log(nameplate_calibration_point) # 1st constant in the explicit solution for eta, given the known values.
    B = np.log(max_weibull_plot_medianrank) # 2nd constant in the explicit solution for eta from the CDF.
    nameplate_eta = int(np.exp((1/nameplate_beta)*(A-B))) # Calculation for nameplate scale factor.
    nmplt = pd.DataFrame(WeibullDPL['dpl']['Y']) #Assign median ranks from dpl to nameplate.
    nmplt.columns = ['Y']
    nmplt_beta = int(50) #assign nameplate beta from user input.
    nmplt_const = nmplt_beta * np.log(nameplate_eta) #calculate the constant from the CDF.
    nmplt['X'] = np.exp((np.log(nmplt['Y'])+nmplt_const)/nmplt_beta) #back calculate the expected production at each probability of the dpl.
    WeibullNMPLT = {
        'nmplt':nmplt,
        'nameplate_eta': nameplate_eta
    }
    return WeibullNMPLT

@st.experimental_memo
def LossesCalculations(WeibullDPL,WeibullRVL,WeibullNMPLT):
    # Calculate the Reliability Losses
    reliability_losses = pd.DataFrame(WeibullDPL['dpl']['X'] - WeibullDPL['WeibullData']['Data'] )
    reliability_losses.columns = ['Reliability Losses']
    r_losses_subset = reliability_losses[reliability_losses['Reliability Losses']>0]
    reliability_losses_sum = int(r_losses_subset.sum().round(0))
    nmplt_efficiency_losses = pd.DataFrame(WeibullNMPLT['nmplt']['X']-WeibullDPL['dpl']['X'])
    nmplt_efficiency_losses.columns = ['Name Plate Efficiency Losses']
    nmplt_efficiency_losses_sum = int(nmplt_efficiency_losses['Name Plate Efficiency Losses'].sum())
    rvl_efficiency_losses = pd.DataFrame(WeibullRVL['rvl']['X']-WeibullDPL['dpl']['X'])
    rvl_efficiency_losses.columns = ['RVL Efficiency Losses']
    rvl_efficiency_losses_sum = int(rvl_efficiency_losses['RVL Efficiency Losses'].sum())
    SystemLosses = {
        'Reliability Losses':reliability_losses_sum,
        'Efficiency Losses (RVL Basis)': rvl_efficiency_losses_sum,
        'Efficiency Losses (NMPLT Basis)': nmplt_efficiency_losses_sum
    }
    return SystemLosses

# Create the Weibull Plot
def weibull_plot2(WeibullDPL,WeibullRVL,WeibullNMPLT,inputsWeibull):
    dpl = WeibullDPL['dpl']
    rvl = WeibullRVL['rvl']
    nmplt = WeibullNMPLT['nmplt']
    Eta = WeibullDPL['Eta']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=WeibullDPL['WeibullData']['Data'],
            y=WeibullDPL['WeibullData']['Plot_MedianRank'],
            mode="markers",
                marker=dict(
                color='black',
                size=4,
                line=dict(
                    color='black',
                    width=1
                )
                )  ,
            name="Production Data"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dpl['X'],
            y=dpl['Y'],
            name="Demonstrated Production Line",
            line=dict(color='goldenrod', width=2,
                              dash='dash')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rvl['X'],
            y=rvl['Y'],
            name="Reduced Variability Line",
            line=dict(color='firebrick', width=2,
                              dash='dash')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nmplt['X'],
            y=nmplt['Y'],
            name="Nameplate",
            line=dict(color='royalblue', width=2,
                              dash='dash')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[Eta[0]],
            y=[np.log(1/(1-0.632))],
            mode="markers",
                marker=dict(
                color='green',
                size=4,
                line=dict(
                    color='green',
                    width=4
                )
                )  ,
            name="Demonstrated Capacity"
        )
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    # This section is to create labels for the Weibull Chart based on Median Rank.
    idy = np.linspace(0,len(WeibullDPL['WeibullData']['Plot_MedianRank'])-1,9).astype(int)
    #idy = np.geomspace(start=2,stop=len(WeibullDPL['WeibullData']['Plot_MedianRank'])-1,num=5,endpoint=True).astype(int)
    idy = idy.tolist()
    Plot_MedianRanks = np.array(WeibullDPL['WeibullData']['Plot_MedianRank'])
    Plot_Y_Labels = np.array(WeibullDPL['WeibullData']['MedianRank'])*100
    levels = [0.01,0.1,0.5,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99,0.995,0.99999]

    fig.update_layout(yaxis = dict(
        tickmode = 'array',
        tickvals = Plot_MedianRanks[idy],
        ticktext = np.around(Plot_Y_Labels[idy],decimals=1)),
        title_text="Weibull Plot",
        title_font_size=24,
        yaxis_title="Percentile",
        xaxis_title=inputsWeibull['analysisColumn'],
        paper_bgcolor='#FCFCFC',
        template='simple_white'
    )

    return st.plotly_chart(fig)
# Create a Histogram for the Weibull sheet.
def WeibullHist(WeibullDPL,inputsWeibull):
    x1 = WeibullDPL['WeibullData']['Data']
    n_bins = bincontroller()
    xmin = min(x1)
    xmax = max(x1)
    s_bins = (xmax-xmin)/n_bins
    fig = ff.create_distplot(hist_data=[x1],group_labels=['Data'],curve_type='normal',bin_size=s_bins,histnorm='probability')
    fig.update_layout(
        title_text="Histogram",
        title_font_size=24,
        yaxis_title="Probability",
        xaxis_title=inputsWeibull['analysisColumn'],
        paper_bgcolor='#FCFCFC',
        template='simple_white'
    )


    return st.plotly_chart(fig)

def bincontroller():
    n_bins = st.slider(value=12,min_value=3,max_value=25,step=1,label='Number of Bins')
    return n_bins

# Create a CUSUM plot.
def cusum_px_plot(WeibullDPL,inputsWeibull):
    cusum = cusumCalcs(WeibullDPL)
    fig = make_subplots(
        rows=2, cols=1,
        column_widths=[1],
        row_heights=[2, 2],
        specs=[[{"type": "scatter"}],
           [{"type": "scatter"}]],
           subplot_titles=("Run Chart","CUSUM Chart"))
    fig.add_trace(
        go.Scatter( x=cusum.index, y=cusum['Data'], name='Data',
                                line=dict(color='black', width=2)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter( x=cusum.index, y=cusum['Moving Average'], name='Moving Average',
                                line=dict(color='firebrick', width=1, dash='dot')),
            row=1, col=1)
    fig.add_trace(
        go.Scatter( x=cusum.index, y=cusum['CUSUM Rolling Deviation'], name='CUSUM Rolling Basis',
                                line=dict(color='black', width=2, dash='dash')),
            row=2, col=1)
    fig.update_layout(
        title_text="Time Series Analysis",
        title_font_size=24,
        yaxis_title=inputsWeibull['analysisColumn'],
        legend_title = "Legend",
        paper_bgcolor='#FCFCFC',
        template='simple_white',
        font=dict(
        size=14,
        color="Black"
    ))
    fig.add_trace(
        go.Scatter( x=cusum.index, y=cusum['CUSUM Nominal Deviation'], name='CUSUM Global Basis',
                                line=dict(color='firebrick', width=2, dash='dot')),
            row=2, col=1)

    return st.plotly_chart(fig)

def cusumCalcs(WeibullDPL):
    cusum = pd.DataFrame(WeibullDPL['data']['Data'])
    cusum.columns  = ['Data']
    rolling_avg = rollingAvgController()
    cusum['Moving Average'] = uniform_filter1d(cusum['Data'], size= int(rolling_avg))
    cusum['Rolling Deviation'] = cusum['Data'] - cusum['Moving Average']
    cusum['Nominal Deviation'] = cusum['Data'] - average(cusum['Data'])
    cusum['CUSUM Rolling Deviation'] = np.cumsum(cusum['Rolling Deviation'])
    cusum['CUSUM Nominal Deviation'] = np.cumsum(cusum['Nominal Deviation'])
    return cusum

def rollingAvgController():
    rolling_avg = st.slider('CUSUM Rolling Average Length', 7, 35, 21) #creates a slider to control the length of the CUSUM.
    return rolling_avg

@st.experimental_memo
def RidgelineDataFilter(RidgelineAnalysisColumn,df):
    RidgelineData = df[RidgelineAnalysisColumn['RidgelineAnalysisColumn']]
    return RidgelineData

def RidgelinePlotTimePeriodSelector():
    with st.form("ridgeline_picker"):
        analysisPeriod = st.selectbox(label="Select the time interval for the Ridgeline Chart (i.e. Year or Quarter)",options=['Year','Quarter','Month'])
        if st.form_submit_button("Select"):
            return analysisPeriod

def RidgelinePlot(RidgelineData,RidgelinePlotTimePeriod,RidgelineAnalysisColumn):
    df = pd.DataFrame(RidgelineData)
    RidgelineTimeIntervalCalcs(df,RidgelinePlotTimePeriod)
    analysisColumn = RidgelineAnalysisColumn['RidgelineAnalysisColumn']

    #This section configures the lower limit and upper limit filter sliders.
    maxVal = max(df[analysisColumn])
    minVal = min(df[analysisColumn])
    midVal = (maxVal-minVal)/3
    lower, upper = Tukey(df[analysisColumn])
    st.caption("The following sliders control the upper and lower bound of the data included in the Ridgeline diagram.  The initial values suggested are selected using Tukey fences to identify outliers.")
    minValue = minValueSelector(minVal,midVal,lower)
    maxValue = maxValueSelector(minValue,maxVal,upper)
    df = df.loc[(df[analysisColumn] >= minValue) & (df[analysisColumn] <= maxValue)]
    periods = {
        'Year':'Year',
        'Quarter':'Year-Quarter',
        'Month':'Year-Month'
    }

    # define the time interval for analysis.
    time_period = periods[RidgelinePlotTimePeriod]
    # define the unique list of indicators for the chart.
    ilist = df[time_period].unique().tolist()
    # define the number of colors required in the chart.
    num = len(ilist)
    # define the column to reference for the plot.
    col = analysisColumn

    # initialize a blank list to be used to build a list of lists.
    rec_ary = []

    # This for loop unstacks the Recovery data by year into a list of lists.
    for x in ilist:
        rec_ary.append(df[df[time_period] == x][col].tolist())
    rec_ary = np.asanyarray(rec_ary,dtype=object) # Convert list of lists to numpy array.

    # the ridge diagram code is "parameterized" sp that once the new data set has been formatted correctly
    # the user just assigns the data to the data variable.
    data = rec_ary
    if num == 1:
        colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 2, colortype='rgb')
    else:
        colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', num, colortype='rgb') # The user currently has to manually set # of colors.

    # This for loop loops through all of the colors and lists that have been defined to create a violin plot for each.
    fig = go.Figure()
    for i, (data_line, color) in enumerate(zip(data, colors)):
        fig.add_trace(
            go.Violin(x=data_line, line_color='black', name=str(ilist[i]), fillcolor=color)
            )

    # use negative ... cuz I'm gonna flip things later
    fig = fig.update_traces(orientation='h', side='negative', width=3, points=False, opacity=1)
    # reverse the (z)-order of the traces
    fig.data = fig.data[::-1]
    # flip the y axis (negative violin is now positive and traces on the top are now on the bottom)
    fig.update_layout(legend_traceorder='reversed', yaxis_autorange='reversed')
    fig.update_layout(
            title='{} by {}'.format(col,time_period),
            xaxis_title=col,
            font= dict(size=18),
            autosize=False,
            width=600,
            height=600,
            margin=dict(
                l=75,
                r=75,
                b=100,
                t=100,
                pad=5
        ),
        paper_bgcolor='#FCFCFC',
        template='simple_white'
    )
    return st.plotly_chart(fig)

def RidgelineTimeIntervalCalcs(df,analysisPeriod):
    # This code extracts the unique list of years contained in the data.
    time_period = []
    df['Year'] = df.index.year
    if analysisPeriod == 'Year':
        time_period = df['Year'].unique().tolist()
    if analysisPeriod == 'Quarter':
        #This section of code extracts the list of quarters.
        df['Quarters'] = df.index.quarter
        df['Year-Quarter'] = df[['Year', 'Quarters']].astype(str).agg('-Q'.join, axis=1)
        time_period=df['Year-Quarter'].unique().tolist()
    if analysisPeriod == 'Month':
        df['Month'] = df.index.month
        df['Month'] = [calendar.month_abbr[x] for x in df['Month']]
        df['Year-Month'] = df[['Year', 'Month']].astype(str).agg('-'.join, axis=1)
    return df

def Tukey(values,multiplier= 2.2):
    values = sorted(values)
    midpoint = int(round(len(values) / 2.0)) # Calculate the midpoint of the sorted list.
    q1 = median(values[:midpoint]) # Lower quartile
    q3 = median(values[midpoint:]) # Upper quartile
    iqr = q3-q1 # calculate the IQR
    lower = q1 - (iqr * multiplier)
    upper = q3 + (iqr * multiplier)
    return lower,upper

if __name__=='__main__':
    st.header('Data Analysis Tools')
    st.sidebar.header('Data Collection')
    st.sidebar.caption("""
    The source excel file for the analyses has several requirements:
    1. The file must include a column of date/time stamps.
    2. The file must include a column containing process data (i.e. Tons per Day).
    """)
    source = sorcery()

    # This checks if the source variable exists in the session state.
    # If it does not, the variable is initialized with the last session state for that variable.
    # This is used in the Sorcery function also reveal user input tasks as steps are completed.
    if not source and 'source' in st.session_state:
        source = st.session_state['source']
    
    # Once Source exists, the next phase of analysis can begin.
    # The above code is necessary to generate a persistent df
    # for analysis mitigating the need to constantly reload data with
    # every change.
    if source:
        st.session_state['source'] = source
        
        df = DataCollector(dSource=source['dSource'],dataSheet=source['dataSheet'],selectedColumns=source['selectedColumns'],dateColumn=source['dateColumn'])

        # Choose which analysis to run.
        analysis = AnalysisPicker()
        # Checking which analysis is chosen and running the associated functions.
        if analysis == "Weibull Production Analysis":
            st.header("Weibull Production Analysis")
            st.write("""
            This section of the analysis suite performs a ***Weibull Production Analysis*** following the method proposed by Roberts and Barringer.
            
            """)
            # These are the date and input controls.
            inputsWeibull = WeibullAnalysisControls(df)

            if not inputsWeibull and 'inputsWeibull' in st.session_state:
                inputsWeibull = st.session_state['inputsWeibull']
            
            if inputsWeibull:
                st.session_state['inputsWeibull'] = inputsWeibull
                WeibullDPL = WeibullAnalysisDPLCalculations(inputsWeibull,df)
                # Defining the beta for the rvl.
                rvl_beta = RVL_sorcery(WeibullDPL)

                if not rvl_beta and 'rvl_beta' in st.session_state:
                    rvl_beta = st.session_state['rvl_beta']

                if rvl_beta:
                    st.session_state['rvl_beta'] = rvl_beta
                    WeibullRVL = WeibullRVLCalculations(WeibullDPL,rvl_beta) # calculate the rvl
                    # Defining the beta of the name plate line.
                    nmplt_beta = NMPLT_sorcery(rvl_beta)

                    if not nmplt_beta and 'nmplt_beta' in st.session_state:
                        nmplt_beta = st.session_state['nmplt_beta']

                    if nmplt_beta:
                        st.session_state['nmplt_beta'] = nmplt_beta

                        WeibullNMPLT = WeibullNMPLTCalculations(WeibullDPL,nmplt_beta) # calculate the name plate line.
                        SystemLosses = LossesCalculations(WeibullDPL,WeibullRVL,WeibullNMPLT)
                        weibull_plot2(WeibullDPL,WeibullRVL,WeibullNMPLT,inputsWeibull) 
                        with st.container():
                            st.write("""
                            **Weibull Analysis Statistics:** 
                            
                            """)
                            col1, col2 = st.columns(spec=2,gap="small")
                            with col1:
                                st.write("**Regression Statistics**")
                                st.write("Eta: " + "{:,}".format(int(WeibullDPL['Eta'])))
                                st.write("Beta: " + str(round(WeibullDPL['Beta'],2)))
                                st.write("R^2: " + str(round(WeibullDPL['Best_Correl_Coef'],3)))
                                st.write("nPoints: " + str(WeibullDPL['nPoints']))
                            with col2:
                                st.write("**Weibull Parameters**")
                                st.write("Nameplate Capacity [TPD]: " + "{:,}".format(WeibullNMPLT['nameplate_eta']))
                                st.write("Reliability Losses: " + "{:,}".format(SystemLosses['Reliability Losses']))
                                st.write("Efficiency Losses (RVL Basis): " + "{:,}".format(SystemLosses['Efficiency Losses (RVL Basis)']))
                                st.write("Efficiency Losses (Name Plate Basis): " + "{:,}".format(SystemLosses['Efficiency Losses (NMPLT Basis)']))
                        WeibullHist(WeibullDPL,inputsWeibull)
                        cusum_px_plot(WeibullDPL,inputsWeibull)
        if analysis == "Ridgeline Diagram":
            st.header("Ridgeline Diagram")
            st.write("""
            This section of the analysis suite generates a Ridgeline Diagram based on user specified time intervals (by Year, by Quarter, or by Month).  
            If more than one year of data is supplied, but either a by quarter or by month image is desired the application will automatically subset
            the time intervals by year and the desired time interval.
            
            """)
            RidgelineAnalysisColumn = RidgelineInputs(df)
            
            if not RidgelineAnalysisColumn and 'RidgelineAnalysisColumn' in st.session_state:
                RidgelineAnalysisColumn = st.session_state['RidgelineAnalysisColumn']
            
            if RidgelineAnalysisColumn:
                st.session_state['RidgelineAnalysisColumn'] = RidgelineAnalysisColumn

                          
                RidgelinePlotTimePeriod = RidgelinePlotTimePeriodSelector()
                if not RidgelinePlotTimePeriod and 'RidgelinePlotTimePeriod' in st.session_state:
                        RidgelinePlotTimePeriod = st.session_state['RidgelinePlotTimePeriod']

                if RidgelinePlotTimePeriod:
                    st.session_state['RidgelinePlotTimePeriod'] = RidgelinePlotTimePeriod

                    RidgelineData = RidgelineDataFilter(RidgelineAnalysisColumn,df)    
                    RidgelinePlot(RidgelineData,RidgelinePlotTimePeriod,RidgelineAnalysisColumn)

            
            
            
            



