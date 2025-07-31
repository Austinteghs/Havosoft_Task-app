import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Havosoft ERP Task Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-text {
        background-color: #DBEAFE;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Havosoft ERP Task Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown("Comprehensive analysis of task data for gap identification and resource optimization")

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Tasks Overview.csv')
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
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.write("Dataset Statistics:")
        st.dataframe(df.describe())

    # Data Preprocessing
    # Convert date columns to datetime
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
    
    # Convert "Total Logged Time" to minutes if it's in HH:MM format
    if 'Total Logged Time' in df.columns:
        def time_to_minutes(time_str):
            try:
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            except:
                return 0
        
        df['Minutes Logged'] = df['Total Logged Time'].apply(time_to_minutes)

    # Dashboard Layout - Using tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Task Status & Completion", "👥 Resource Allocation", "⏱️ Time Analysis", "🔍 Gap Analysis"])
    
    # Tab 1: Task Status and Completion
    with tab1:
        st.markdown('<div class="sub-header">Task Status & Completion Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Task Status Distribution
            st.subheader("Task Status Distribution")
            status_counts = df['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            fig = px.pie(
                status_counts, 
                values='Count', 
                names='Status',
                title='Task Status Distribution',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # On-Time Completion Rate
            st.subheader("On-Time Completion Analysis")
            if 'Finished on time?' in df.columns:
                ontime_counts = df['Finished on time?'].value_counts().reset_index()
                ontime_counts.columns = ['Finished on time', 'Count']
                
                fig = px.bar(
                    ontime_counts,
                    x='Finished on time',
                    y='Count',
                    title='Tasks Finished On Time vs. Delayed',
                    color='Finished on time',
                    color_discrete_sequence=['#4CAF50', '#F44336']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("On-time completion data not available in the dataset.")
        
        # Task completion over time
        st.subheader("Task Completion Timeline")
        if 'Due Date' in df.columns and 'Status' in df.columns:
            # Filter for completed tasks
            completed_tasks = df[df['Status'] == 'Complete'].copy()
            if not completed_tasks.empty:
                completed_tasks['Due Month'] = completed_tasks['Due Date'].dt.month
                completed_tasks['Due Month Name'] = completed_tasks['Due Date'].dt.month_name()
                
                monthly_completion = completed_tasks.groupby('Due Month Name').size().reset_index()
                monthly_completion.columns = ['Month', 'Completed Tasks']
                
                # Sort months chronologically
                months_order = {month: i for i, month in enumerate(calendar.month_name[1:])}
                monthly_completion['Month_Num'] = monthly_completion['Month'].map(months_order)
                monthly_completion = monthly_completion.sort_values('Month_Num').drop('Month_Num', axis=1)
                
                fig = px.line(
                    monthly_completion, 
                    x='Month', 
                    y='Completed Tasks',
                    markers=True,
                    title='Task Completion by Month',
                    line_shape='linear'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completed tasks found in the dataset.")
        else:
            st.info("Due Date or Status column not available in the dataset.")
            
        # Task Complexity Analysis (using proxy measures)
        st.subheader("Task Complexity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Total comments' in df.columns:
                # Group tasks by number of comments
                comment_bins = [0, 1, 3, 5, 10, 100]
                comment_labels = ['No comments', '1-3 comments', '4-5 comments', '6-10 comments', '10+ comments']
                
                df['Comment Range'] = pd.cut(pd.to_numeric(df['Total comments'], errors='coerce'), 
                                          bins=comment_bins, 
                                          labels=comment_labels, 
                                          include_lowest=True)
                
                comment_dist = df['Comment Range'].value_counts().reset_index()
                comment_dist.columns = ['Comment Range', 'Count']
                
                fig = px.bar(
                    comment_dist,
                    x='Comment Range',
                    y='Count',
                    title='Task Distribution by Number of Comments',
                    color='Comment Range'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Comment data not available.")
                
        with col2:
            if 'Total attachments added' in df.columns:
                # Group tasks by number of attachments
                attachment_bins = [0, 1, 3, 5, 10, 100]
                attachment_labels = ['No attachments', '1-3 attachments', '4-5 attachments', '6-10 attachments', '10+ attachments']
                
                df['Attachment Range'] = pd.cut(pd.to_numeric(df['Total attachments added'], errors='coerce'), 
                                             bins=attachment_bins, 
                                             labels=attachment_labels, 
                                             include_lowest=True)
                
                attachment_dist = df['Attachment Range'].value_counts().reset_index()
                attachment_dist.columns = ['Attachment Range', 'Count']
                
                fig = px.bar(
                    attachment_dist,
                    x='Attachment Range',
                    y='Count',
                    title='Task Distribution by Number of Attachments',
                    color='Attachment Range'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Attachment data not available.")
    
    # Tab 2: Resource Allocation Analysis
    with tab2:
        st.markdown('<div class="sub-header">Resource Allocation Analysis</div>', unsafe_allow_html=True)
        
        if 'Assigned to' in df.columns:
            # Task distribution by assignee
            st.subheader("Task Distribution by Team Member")
            
            staff_task_counts = df['Assigned to'].value_counts().reset_index()
            staff_task_counts.columns = ['Staff Member', 'Number of Tasks']
            
            # Sort by number of tasks for better visualization
            staff_task_counts = staff_task_counts.sort_values('Number of Tasks', ascending=False)
            
            # Limit to top 15 for readability
            top_staff = staff_task_counts.head(15)
            
            fig = px.bar(
                top_staff,
                x='Staff Member',
                y='Number of Tasks',
                title='Task Distribution Across Team Members (Top 15)',
                color='Number of Tasks',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Staff workload analysis
            st.subheader("Staff Workload Analysis")
            
            if 'Status' in df.columns and 'Assigned to' in df.columns:
                # Create a staff workload DataFrame
                staff_workload = df.groupby('Assigned to')['Status'].value_counts().unstack().fillna(0)
                
                # Reset index for plotting
                staff_workload = staff_workload.reset_index()
                
                # Sort by total tasks
                staff_workload['Total'] = staff_workload.sum(axis=1)
                staff_workload = staff_workload.sort_values('Total', ascending=False).head(10)
                
                # Prepare data for stacked bar chart
                status_cols = [col for col in staff_workload.columns if col not in ['Assigned to', 'Total']]
                
                fig = go.Figure()
                
                for status in status_cols:
                    fig.add_trace(go.Bar(
                        name=status,
                        x=staff_workload['Assigned to'],
                        y=staff_workload[status],
                    ))
                
                fig.update_layout(
                    title='Workload Distribution by Status (Top 10 Staff)',
                    xaxis_title='Staff Member',
                    yaxis_title='Number of Tasks',
                    barmode='stack',
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Staff completion rate
            st.subheader("Task Completion Rate by Staff")
            
            if 'Status' in df.columns and 'Assigned to' in df.columns:
                # Get completion rate for staff with more than 5 tasks
                staff_completion = df.groupby('Assigned to').agg({
                    'Name': 'count',
                    'Status': lambda x: (x == 'Complete').sum() / len(x) * 100
                }).reset_index()
                
                staff_completion.columns = ['Staff Member', 'Total Tasks', 'Completion Rate (%)']
                staff_completion = staff_completion[staff_completion['Total Tasks'] > 3].sort_values('Completion Rate (%)', ascending=False)
                
                fig = px.bar(
                    staff_completion,
                    x='Staff Member',
                    y='Completion Rate (%)',
                    title='Task Completion Rate by Staff Member (For staff with >3 tasks)',
                    color='Completion Rate (%)',
                    color_continuous_scale='RdYlGn',
                    text='Completion Rate (%)'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate average completion rate
                avg_completion = staff_completion['Completion Rate (%)'].mean()
                
                # Highlight staff with lower than average completion rates
                st.markdown('<div class="insight-text">', unsafe_allow_html=True)
                st.write(f"**Team Average Completion Rate: {avg_completion:.1f}%**")
                
                below_avg = staff_completion[staff_completion['Completion Rate (%)'] < avg_completion]
                if not below_avg.empty:
                    st.write("**Staff members with below-average completion rates:**")
                    for idx, row in below_avg.iterrows():
                        st.write(f"- {row['Staff Member']}: {row['Completion Rate (%)']:.1f}% ({row['Total Tasks']} tasks)")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Assignment data not available in the dataset.")
    
    # Tab 3: Time Analysis
    with tab3:
        st.markdown('<div class="sub-header">Time Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Task Duration Analysis
            st.subheader("Task Duration Analysis")
            
            if 'Task Duration (Days)' in df.columns:
                # Create duration bins for better visualization
                duration_bins = [-1, 0, 1, 7, 14, 30, 90]
                duration_labels = ['Same day', '1 day', '2-7 days', '8-14 days', '15-30 days', '30+ days']
                
                df['Duration Category'] = pd.cut(df['Task Duration (Days)'], 
                                               bins=duration_bins, 
                                               labels=duration_labels, 
                                               include_lowest=True)
                
                duration_dist = df['Duration Category'].value_counts().reset_index()
                duration_dist.columns = ['Duration', 'Count']
                
                # Sort by duration for better interpretation
                duration_map = {label: i for i, label in enumerate(duration_labels)}
                duration_dist['sort_key'] = duration_dist['Duration'].map(duration_map)
                duration_dist = duration_dist.sort_values('sort_key').drop('sort_key', axis=1)
                
                fig = px.bar(
                    duration_dist,
                    x='Duration',
                    y='Count',
                    title='Task Distribution by Duration',
                    color='Duration',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Average task duration by month
                st.subheader("Average Task Duration by Month")
                
                if 'Start Month' in df.columns:
                    month_duration = df.groupby('Start Month')['Task Duration (Days)'].mean().reset_index()
                    
                    # Sort months chronologically
                    months_order = {month: i for i, month in enumerate(calendar.month_name[1:])}
                    month_duration['Month_Num'] = month_duration['Start Month'].map(months_order)
                    month_duration = month_duration.sort_values('Month_Num').drop('Month_Num', axis=1)
                    
                    fig = px.bar(
                        month_duration,
                        x='Start Month',
                        y='Task Duration (Days)',
                        title='Average Task Duration by Month',
                        color='Task Duration (Days)',
                        text='Task Duration (Days)'
                    )
                    fig.update_traces(texttemplate='%{text:.1f} days', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Task duration data could not be calculated.")
        
        with col2:
            # Time Logged Analysis
            st.subheader("Time Logged Analysis")
            
            if 'Minutes Logged' in df.columns:
                # Create time bins for better visualization
                time_bins = [0, 30, 60, 120, 240, 480, 10000]
                time_labels = ['< 30 min', '30-60 min', '1-2 hours', '2-4 hours', '4-8 hours', '8+ hours']
                
                df['Time Category'] = pd.cut(df['Minutes Logged'], 
                                            bins=time_bins, 
                                            labels=time_labels, 
                                            include_lowest=True)
                
                time_dist = df['Time Category'].value_counts().reset_index()
                time_dist.columns = ['Time Logged', 'Count']
                
                # Sort by time for better interpretation
                time_map = {label: i for i, label in enumerate(time_labels)}
                time_dist['sort_key'] = time_dist['Time Logged'].map(time_map)
                time_dist = time_dist.sort_values('sort_key').drop('sort_key', axis=1)
                
                fig = px.bar(
                    time_dist,
                    x='Time Logged',
                    y='Count',
                    title='Task Distribution by Time Logged',
                    color='Time Logged',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Staff Time Investment
                st.subheader("Staff Time Investment Analysis")
                
                if 'Assigned to' in df.columns:
                    staff_time = df.groupby('Assigned to')['Minutes Logged'].agg(['sum', 'mean', 'count']).reset_index()
                    staff_time.columns = ['Staff Member', 'Total Minutes', 'Average Minutes per Task', 'Task Count']
                    
                    # Convert to hours for readability
                    staff_time['Total Hours'] = staff_time['Total Minutes'] / 60
                    staff_time['Average Hours per Task'] = staff_time['Average Minutes per Task'] / 60
                    
                    # Sort by total time spent
                    staff_time = staff_time.sort_values('Total Hours', ascending=False).head(10)
                    
                    fig = px.bar(
                        staff_time,
                        x='Staff Member',
                        y='Total Hours',
                        title='Total Hours Logged by Staff (Top 10)',
                        color='Total Hours',
                        text='Total Hours'
                    )
                    fig.update_traces(texttemplate='%{text:.1f} hrs', textposition='outside')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Time logged data could not be calculated.")
    
    # Tab 4: Gap Analysis
    with tab4:
        st.markdown('<div class="sub-header">Gap Analysis & Insights</div>', unsafe_allow_html=True)
        
        # Planned vs. Actual Completion Analysis
        st.subheader("Planned vs. Actual Task Completion")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tasks = len(df)
            st.metric("Total Tasks", f"{total_tasks}")
        
        with col2:
            if 'Status' in df.columns:
                completed_tasks = df[df['Status'] == 'Complete'].shape[0]
                completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            else:
                st.metric("Completion Rate", "N/A")
        
        with col3:
            if 'Finished on time?' in df.columns:
                on_time_tasks = df[df['Finished on time?'] == 'Yes'].shape[0]
                on_time_rate = (on_time_tasks / completed_tasks) * 100 if completed_tasks > 0 else 0
                st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
            else:
                st.metric("On-Time Rate", "N/A")
        
        with col4:
            if 'Task Duration (Days)' in df.columns:
                avg_duration = df['Task Duration (Days)'].mean()
                st.metric("Avg. Duration", f"{avg_duration:.1f} days")
            else:
                st.metric("Avg. Duration", "N/A")
        
        # Visualize task delays if data available
        if 'Start Date' in df.columns and 'Due Date' in df.columns and 'Status' in df.columns:
            st.subheader("Task Delay Analysis")
            
            # Calculate delay for completed tasks
            completed_df = df[df['Status'] == 'Complete'].copy()
            
            if not completed_df.empty and 'Finished on time?' in completed_df.columns:
                delayed_tasks = completed_df[completed_df['Finished on time?'] == 'No']
                
                if not delayed_tasks.empty:
                    # Analyze delays by assignee
                    if 'Assigned to' in delayed_tasks.columns:
                        delay_by_assignee = delayed_tasks.groupby('Assigned to').size().reset_index()
                        delay_by_assignee.columns = ['Staff Member', 'Delayed Tasks']
                        delay_by_assignee = delay_by_assignee.sort_values('Delayed Tasks', ascending=False).head(10)
                        
                        fig = px.bar(
                            delay_by_assignee,
                            x='Staff Member',
                            y='Delayed Tasks',
                            title='Staff Members with Delayed Tasks (Top 10)',
                            color='Delayed Tasks',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("Great news! No delayed tasks found in the dataset.")
            else:
                st.info("Delay analysis could not be performed due to missing data.")
        
        # Resource Gap Analysis
        st.subheader("Resource Gap Analysis")
        
        if 'Assigned to' in df.columns and 'Status' in df.columns:
            # Calculate workload distribution and identify potential gaps
            staff_workload = df.groupby('Assigned to').agg({
                'Name': 'count',
                'Status': lambda x: (x == 'Complete').sum() / len(x) * 100 if len(x) > 0 else 0,
                'Minutes Logged': 'sum'
            }).reset_index()
            
            staff_workload.columns = ['Staff Member', 'Task Count', 'Completion Rate (%)', 'Total Minutes']
            staff_workload['Hours Logged'] = staff_workload['Total Minutes'] / 60
            
            # Calculate average metrics
            avg_task_count = staff_workload['Task Count'].mean()
            avg_completion_rate = staff_workload['Completion Rate (%)'].mean()
            
            # Identify potential overloaded staff (high task count, lower completion rate)
            potentially_overloaded = staff_workload[
                (staff_workload['Task Count'] > avg_task_count * 1.5) & 
                (staff_workload['Completion Rate (%)'] < avg_completion_rate)
            ]
            
            if not potentially_overloaded.empty:
                st.markdown('<div class="insight-text">', unsafe_allow_html=True)
                st.write("**Potential Resource Gaps Detected:**")
                st.write("The following staff members may be overloaded based on task count and completion rates:")
                
                for idx, row in potentially_overloaded.iterrows():
                    st.write(f"- **{row['Staff Member']}**: {row['Task Count']} tasks, {row['Completion Rate (%)']:.1f}% completion rate, {row['Hours Logged']:.1f} hours logged")
                
                st.write("**Recommendation:** Consider redistributing tasks or providing additional resources.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("No significant resource gaps detected based on task distribution and completion rates.")
            
            # Task Distribution Visualization
            fig = px.scatter(
                staff_workload,
                x='Task Count',
                y='Completion Rate (%)',
                size='Hours Logged',
                hover_name='Staff Member',
                title='Staff Workload Analysis: Task Count vs. Completion Rate',
                color='Completion Rate (%)',
                color_continuous_scale='RdYlGn',
                size_max=50
            )
            
            fig.add_shape(
                type="rect",
                x0=avg_task_count * 1.5,
                y0=0,
                x1=staff_workload['Task Count'].max() + 1,
                y1=avg_completion_rate,
                line=dict(color="Red"),
                fillcolor="Red",
                opacity=0.1,
                name="Potential Overload Zone"
            )
            
            fig.add_hline(
                y=avg_completion_rate,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Avg Completion Rate: {avg_completion_rate:.1f}%"
            )
            
            fig.add_vline(
                x=avg_task_count * 1.5,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"High Task Count: >{avg_task_count * 1.5:.0f}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Insights and Recommendations
        st.subheader("Key Insights & Recommendations")
        
        st.markdown('<div class="insight-text">', unsafe_allow_html=True)
        st.write("**Overall Task Management Insights:**")
        
        # Generate insights based on available data
        insights = []
        recommendations = []
        
        # Task completion insights
        if 'Status' in df.columns:
            completion_rate = (df[df['Status'] == 'Complete'].shape[0] / len(df)) * 100 if len(df) > 0 else 0
            if completion_rate < 70:
                insights.append(f"The overall task completion rate is {completion_rate:.1f}%, which is below the ideal target of 80%+.")
                recommendations.append("Implement regular progress tracking and milestone reviews to improve completion rates.")
            else:
                insights.append(f"Strong overall task completion rate of {completion_rate:.1f}%.")
        
        # Time estimation insights
        if 'Finished on time?' in df.columns and 'Status' in df.columns:
            completed_tasks = df[df['Status'] == 'Complete']
            if not completed_tasks.empty:
                ontime_rate = (completed_tasks[completed_tasks['Finished on time?'] == 'Yes'].shape[0] / len(completed_tasks)) * 100
                if ontime_rate < 85:
                    insights.append(f"Only {ontime_rate:.1f}% of completed tasks were finished on time, suggesting time estimation challenges.")
                    recommendations.append("Review time estimation processes and consider adding buffer time for complex tasks.")
                else:
                    insights.append(f"Good on-time completion rate of {ontime_rate:.1f}% for completed tasks.")
        
        # Resource allocation insights
        if 'Assigned to' in df.columns:
            task_distribution = df['Assigned to'].value_counts()
            max_tasks = task_distribution.max()
            min_tasks = task_distribution.min()
            if max_tasks > min_tasks * 3 and len(task_distribution) > 5:
                insights.append("Significant imbalance in task distribution across team members.")
                recommendations.append("Review workload distribution to ensure balanced allocation of tasks.")
        
        # Output insights and recommendations
        for insight in insights:
            st.write(f"- {insight}")
        
        st.write("**Recommendations:**")
        if recommendations:
            for recommendation in recommendations:
                st.write(f"- {recommendation}")
        else:
            st.write("- Continue monitoring current processes as they appear to be working effectively.")
            st.write("- Consider implementing predictive analytics for future resource planning.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">Havosoft ERP Task Analytics Dashboard | Created with Streamlit | Data Analysis Date: July 31, 2025</div>', unsafe_allow_html=True)

else:
    st.error("Failed to load data. Please check the file path and format.")