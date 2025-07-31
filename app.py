import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Set page configuration
st.set_page_config(
    page_title="Havosoft ERP Task Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("Havosoft ERP Task Analytics Dashboard")
st.markdown("Comprehensive analysis of task data for gap identification and resource optimization")

# Load data function with better error handling
@st.cache_data
def load_data():
    try:
        # Define the file path
        file_path = 'Tasks Overview.csv'
        
        # Try to read with explicit encoding
        df = pd.read_csv(file_path)
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Display data overview
    with st.expander("📋 Data Overview"):
        st.write("Sample of the task data:")
        st.dataframe(df.head())
        
        st.write("Dataset Information:")
        st.text(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.text(f"Column names: {', '.join(df.columns.tolist())}")
        
        st.write("Dataset Statistics:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe())

    # Data Preprocessing
    # Convert date columns to datetime
    try:
        date_cols = ['Start Date', 'Due Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create new features for analysis
        if 'Start Date' in df.columns and 'Due Date' in df.columns:
            # Calculate task duration in days
            df['Task Duration (Days)'] = (df['Due Date'] - df['Start Date']).dt.days
            
            # Extract month and week information
            df['Start Month'] = df['Start Date'].dt.month_name()
            df['Due Month'] = df['Due Date'].dt.month_name()
            df['Start Week'] = df['Start Date'].dt.isocalendar().week
            df['Due Week'] = df['Due Date'].dt.isocalendar().week
    except Exception as e:
        st.warning(f"Issue processing date columns: {e}")
    
    # Convert "Total Logged Time" to minutes if it's in HH:MM format
    try:
        if 'Total Logged Time' in df.columns:
            def time_to_minutes(time_str):
                try:
                    if pd.isna(time_str):
                        return 0
                    hours, minutes = map(int, str(time_str).split(':'))
                    return hours * 60 + minutes
                except:
                    return 0
            
            df['Minutes Logged'] = df['Total Logged Time'].apply(time_to_minutes)
    except Exception as e:
        st.warning(f"Issue processing time data: {e}")

    # Dashboard Layout - Using tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Task Status", "👥 Resource Allocation", "⏱️ Time Analysis", "🔍 Gap Analysis"])
    
    # Tab 1: Task Status and Completion
    with tab1:
        st.header("Task Status & Completion Analysis")
        
        # Basic task metrics
        total_tasks = len(df)
        st.metric("Total Tasks", total_tasks)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Task Status Distribution
            if 'Status' in df.columns:
                st.subheader("Task Status Distribution")
                try:
                    status_counts = df['Status'].value_counts()
                    fig = px.pie(
                        names=status_counts.index,
                        values=status_counts.values,
                        title='Task Status Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating status chart: {e}")
            
        with col2:
            # On-Time Completion Rate
            if 'Finished on time?' in df.columns:
                st.subheader("On-Time Completion Analysis")
                try:
                    ontime_counts = df['Finished on time?'].value_counts()
                    fig = px.bar(
                        x=ontime_counts.index,
                        y=ontime_counts.values,
                        title='Tasks Finished On Time vs. Delayed'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating on-time chart: {e}")
            else:
                st.info("On-time completion data not available in the dataset.")
    
    # Tab 2: Resource Allocation Analysis
    with tab2:
        st.header("Resource Allocation Analysis")
        
        if 'Assigned to' in df.columns:
            # Task distribution by assignee
            st.subheader("Task Distribution by Team Member")
            
            try:
                staff_task_counts = df['Assigned to'].value_counts().reset_index()
                staff_task_counts.columns = ['Staff Member', 'Number of Tasks']
                
                # Sort by number of tasks for better visualization
                staff_task_counts = staff_task_counts.sort_values('Number of Tasks', ascending=False)
                
                # Limit to top 10 for readability
                top_staff = staff_task_counts.head(10)
                
                fig = px.bar(
                    top_staff,
                    x='Staff Member',
                    y='Number of Tasks',
                    title='Task Distribution Across Team Members (Top 10)'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating staff task distribution chart: {e}")
        else:
            st.info("Assignment data not available in the dataset.")
    
    # Tab 3: Time Analysis
    with tab3:
        st.header("Time Analysis")
        
        if 'Task Duration (Days)' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Task Duration Analysis
                st.subheader("Task Duration Analysis")
                
                try:
                    # Average duration
                    avg_duration = df['Task Duration (Days)'].mean()
                    st.metric("Average Task Duration", f"{avg_duration:.1f} days")
                    
                    # Create duration histogram
                    fig = px.histogram(
                        df,
                        x='Task Duration (Days)',
                        title='Distribution of Task Duration',
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating duration chart: {e}")
            
            with col2:
                # Time Logged Analysis
                if 'Minutes Logged' in df.columns:
                    st.subheader("Time Logged Analysis")
                    
                    try:
                        # Average time logged
                        avg_minutes = df['Minutes Logged'].mean()
                        st.metric("Average Time Logged", f"{avg_minutes:.1f} minutes")
                        
                        # Create time logged histogram
                        fig = px.histogram(
                            df,
                            x='Minutes Logged',
                            title='Distribution of Time Logged',
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating time logged chart: {e}")
        else:
            st.info("Task duration data not available.")
    
    # Tab 4: Gap Analysis
    with tab4:
        st.header("Gap Analysis & Insights")
        
        # Create metrics row
        if 'Status' in df.columns:
            try:
                # Calculate completion rate
                completed_tasks = df[df['Status'] == 'Complete'].shape[0] if 'Complete' in df['Status'].values else 0
                completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
                
                # Key insights
                st.subheader("Key Insights")
                
                if completion_rate < 70:
                    st.warning("The overall task completion rate is below the ideal target of 80%+.")
                    st.markdown("**Recommendation:** Implement regular progress tracking and milestone reviews to improve completion rates.")
                else:
                    st.success("Strong overall task completion rate.")
            except Exception as e:
                st.error(f"Error in gap analysis: {e}")
        
        # Resource allocation insights
        if 'Assigned to' in df.columns:
            st.subheader("Resource Allocation Insights")
            
            try:
                task_distribution = df['Assigned to'].value_counts()
                max_tasks = task_distribution.max()
                min_tasks = task_distribution.min()
                
                if len(task_distribution) > 5 and max_tasks > min_tasks * 3:
                    st.warning("There appears to be an imbalance in task distribution across team members.")
                    st.markdown("**Recommendation:** Review workload distribution to ensure balanced allocation of tasks.")
            except Exception as e:
                st.error(f"Error in resource allocation analysis: {e}")
else:
    st.error("Failed to load data. Please check the file path and format.")
    st.markdown("Make sure the CSV file is located at: `/workspace/uploads/Tasks Overview.csv`")
