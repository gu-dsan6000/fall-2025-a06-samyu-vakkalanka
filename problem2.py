from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, input_file_name, regexp_extract, col, count, min, max, when
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import argparse

def create_spark_session(master_url):
    """Create a Spark session."""

    spark = (
        SparkSession.builder
        .appName("Log_Level_Distribution_Cluster")

         # Cluster Configuration
        .master(master_url)  # Connect to Spark cluster

        # Memory Configuration
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")

        # Executor Configuration
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")  # Use all available cores across cluster

        # S3 Configuration - Use S3A for AWS S3 access
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

        # Performance settings for cluster execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Arrow optimization for Pandas conversion
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")


        .getOrCreate()
    )
    return spark

def parse_logs_with_ids(logs_df):
    """Parse logs and extract cluster/application/container IDs."""
    
    # Add file path and extract IDs from path
    logs_with_path = logs_df.withColumn('file_path', input_file_name())
    
    # Extract application and container IDs from file path
    logs_with_ids = logs_with_path.withColumn(
        'application_id',
        regexp_extract('file_path', r'application_(\d+_\d+)', 0)
    ).withColumn(
        'container_id',
        regexp_extract('file_path', r'(container_\d+_\d+_\d+_\d+)', 1)
    )
    
    # Parse the log content itself
    parsed_logs = logs_with_ids.select(
        regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp_str'),
        regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('log_level'),
        regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)\s+([^:]+):', 2).alias('component'),
        col('value').alias('message'),
        col('application_id'),
        col('container_id'),
        col('file_path')
    )
    parsed_logs = parsed_logs.withColumn(
        'timestamp',
        when(col('timestamp_str') != '', to_timestamp('timestamp_str', 'yy/MM/dd HH:mm:ss'))
        .otherwise(None)
    )
    
    # Extract cluster_id from application_id (the first part before the underscore)
    parsed_logs = parsed_logs.withColumn(
        'cluster_id',
        regexp_extract('application_id', r'application_(\d+)_\d+', 1)
    )
    
    # Extract app_number from application_id (the second part after the underscore)
    parsed_logs = parsed_logs.withColumn(
        'app_number',
        regexp_extract('application_id', r'application_\d+_(\d+)', 1)
    )
    
    return parsed_logs

def create_timeline(parsed_logs):
    """Create timeline dataset with start/end times for each application."""
    
    # Group by cluster and application to get start/end times
    timeline = parsed_logs.groupBy('cluster_id', 'application_id', 'app_number').agg(
        min('timestamp').alias('start_time'),
        max('timestamp').alias('end_time')
    ).orderBy('cluster_id', 'app_number')

    output_file = 'data/output/problem2_timeline.csv'
    timeline.toPandas().to_csv(output_file, index=False)
    return timeline

def create_cluster_summary(timeline):
    """Create cluster summary with aggregated statistics."""
    
    cluster_summary = timeline.groupBy('cluster_id').agg(
        count('application_id').alias('num_applications'),
        min('start_time').alias('cluster_first_app'),
        max('end_time').alias('cluster_last_app')
    ).orderBy(col('num_applications').desc())
    
    output_file = 'data/output/problem2_cluster_summary.csv'
    cluster_summary.toPandas().to_csv(output_file, index=False)
    
    return cluster_summary

def get_stats(cluster_summary):
    """Create summary statistics text file."""
    # Collect data for analysis
    cluster_data = cluster_summary.orderBy(col('num_applications').desc()).collect()
    
    total_clusters = len(cluster_data)
    total_applications = sum(row['num_applications'] for row in cluster_data)
    avg_apps_per_cluster = total_applications / total_clusters if total_clusters > 0 else 0
    
    # Build stats text
    stats_lines = []
    stats_lines.append(f"Total unique clusters: {total_clusters}")
    stats_lines.append(f"Total applications: {total_applications}")
    stats_lines.append(f"Average applications per cluster: {avg_apps_per_cluster:.2f}")
    stats_lines.append("")
    stats_lines.append("Most heavily used clusters:")
    
    for row in cluster_data:
        stats_lines.append(f"  Cluster {row['cluster_id']}: {row['num_applications']} applications")
    
    # Write to file
    output_file = 'data/output/problem2_stats.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(stats_lines))
        
def create_bar_chart(df):
    """Create bar chart showing number of applications per cluster."""    
    # Sort by number of applications descending
    df = df.sort_values('num_applications', ascending=False)
    
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Create bar chart using seaborn
    sns.barplot(
        data=df,
        x='cluster_id',
        y='num_applications',
        ax=ax
    )
    
    # Add value labels on top of bars
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(
            i,
            row['num_applications'] + 1,
            str(row['num_applications']),
            ha='center',
            va='bottom'
        )
    
    # Set labels and title
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Applications')
    ax.set_title('Number of Applications per Cluster')
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    output_file = 'data/output/problem2_bar_chart.png'
    plt.savefig(output_file)
    plt.close()
    
def create_density_plot(timeline_df, cluster_summary_df):
    """Create density plot showing job duration distribution for the largest cluster."""
    # Find the cluster with the most applications
    largest_cluster = cluster_summary_df.sort_values('num_applications', ascending=False).iloc[0]
    largest_cluster_id = largest_cluster['cluster_id']
    
    # Filter timeline for largest cluster
    cluster_data = timeline_df[timeline_df['cluster_id'] == largest_cluster_id].copy()
    
    # Convert timestamps to datetime
    cluster_data['start_time'] = pd.to_datetime(cluster_data['start_time'], format='%y/%m/%d %H:%M:%S')
    cluster_data['end_time'] = pd.to_datetime(cluster_data['end_time'], format='%y/%m/%d %H:%M:%S')
    
    # Calculate duration in minutes
    cluster_data['duration_minutes'] = (
        cluster_data['end_time'] - cluster_data['start_time']
    ).dt.total_seconds() / 60
    
    # Remove any invalid durations (negative or zero)
    cluster_data = cluster_data[cluster_data['duration_minutes'] > 0]
    
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Create histogram with KDE overlay using seaborn
    sns.histplot(
        data=cluster_data,
        x='duration_minutes',
        bins=30,
        kde=True,
        ax=ax
    )
    
    # Set log scale on x-axis to handle skewed data
    ax.set_xscale('log')
    
    # Set labels and title
    ax.set_xlabel('Job Duration (minutes, log scale)')
    ax.set_ylabel('Frequency')
    ax.set_title(
        f'Job Duration Distribution for Cluster {largest_cluster_id} (n={len(cluster_data)})'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = 'data/output/problem2_density_plot.png'
    plt.savefig(output_file)
    
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process Spark cluster logs')
    parser.add_argument('master_url', nargs='?', help='Spark master URL (e.g., spark://MASTER_IP:7077)')
    parser.add_argument('--net-id', type=str, help='Net ID for S3 bucket name')
    parser.add_argument('--skip-spark', action='store_true', help='Skip Spark processing and use existing CSVs')
    
    args = parser.parse_args()
    if args.skip_spark:
        try:
            timeline_file = 'data/output/problem2_timeline.csv'
            timeline = pd.read_csv(timeline_file)
            timeline['start_time'] = pd.to_datetime(timeline['start_time'], format="%Y-%m-%d %H:%M:%S")
            timeline['end_time'] = pd.to_datetime(timeline['end_time'], format="%Y-%m-%d %H:%M:%S")

            summary_file = 'data/output/problem2_cluster_summary.csv'
            cluster_summary = pd.read_csv(summary_file)
            cluster_summary['cluster_first_app'] = pd.to_datetime(cluster_summary['cluster_first_app'], format="%Y-%m-%d %H:%M:%S")
            cluster_summary['cluster_last_app'] = pd.to_datetime(cluster_summary['cluster_last_app'], format="%Y-%m-%d %H:%M:%S")

            create_bar_chart(cluster_summary)
            create_density_plot(timeline, cluster_summary)
            return 0
        
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return 1
    
    if args.master_url:
        master_url = args.master_url
    else:
        # Try to get from environment variable
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("❌ Error: Master URL not provided")
            print("Usage: python problem2.py spark://MASTER_IP:7077 --net-id YOUR-NET-ID")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            print("   or: python problem2.py --skip-spark (to regenerate visualizations only)")
            return 1
    spark = create_spark_session(master_url)
    if args.net_id:
        net_id = args.net_id
    else:
        net_id = os.getenv("YOUR_NET_ID", "spv15")
    try:
        # Load all log files
        s3_path = f"s3a://{net_id}-assignment-spark-cluster-logs/data/*/*.log"
        logs_df = spark.read.text(s3_path)
        success = True
    
    except Exception as e:
        print(f"Error loading log files: {str(e)}")
        success = False

    try:
        # Parse logs and extract all IDs
        parsed_logs = parse_logs_with_ids(logs_df)
    except Exception as e:
        print(f"Error loading parsing log files: {str(e)}")
        success = False
    
    try:
        # Get timeline
        timeline = create_timeline(parsed_logs)
    except Exception as e:
        print(f"Error creating timeline: {str(e)}")
        success = False
    
    try:
        # Get cluster summary
        cluster_summary = create_cluster_summary(timeline)
    except Exception as e:
        print(f"Error creating cluster summary: {str(e)}")
        success = False

    try:
        # Get stats
        get_stats(cluster_summary)
    except Exception as e:
        print(f"Error creating stats file: {str(e)}")
        success = False
    
    try:
        # Get bar chart
        cluster_summary_df = cluster_summary.toPandas()
        create_bar_chart(cluster_summary)
    except Exception as e:
        print(f"Error creating bar chart: {str(e)}")
        success = False
    
    try:
        # Get density plot
        timeline_df = timeline.toPandas()
        create_density_plot(timeline_df, cluster_summary_df)
    except Exception as e:
        print(f"Error creating density plot: {str(e)}")
        success = False

    # Clean up
    spark.stop()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())