import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col
import pandas as pd

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

def parse_logs(log_text):
    # Parse logs into df (code from hint section)
    parsed_logs = log_text.select(
        regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
        regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('level'),
        regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)\s+([^:]+):', 2).alias('component'),
        col('value').alias('message')
    )
    parsed_logs = parsed_logs.filter(col("level").isNotNull())
    parsed_logs = parsed_logs.filter(col('level') != '')
    return parsed_logs

def level_counts(parsed_logs):
    # Code adapted from DATASET_OVERVIEW.md
    # Analyze log levels
    log_level_counts = (
        parsed_logs
        .groupBy('level')
        .count()
    )
    output_file = 'data/output/problem1_counts_local.csv'
    log_level_counts.toPandas().to_csv(output_file, index=False)

def get_random(parsed_logs):
    # Get random sample of 10 rows from parsed df
    random_sample = parsed_logs[['message', 'level']].sample(withReplacement=False, fraction=(10/parsed_logs.count()))
    random_sample = random_sample.withColumnRenamed("message", "log_entry").withColumnRenamed("level", "log_level").limit(10)
    output_file = 'data/output/problem1_sample_local.csv'
    random_sample.toPandas().to_csv(output_file, index=False)

def get_summary(total_lines, total_with_levels, unique_levels, parsed_logs):
    # Construct summary from log data
    # Group parsed logs by level and count 
    level_distribution = (
        parsed_logs
        .groupBy('level')
        .count()
        .orderBy('count', ascending=False)
        .collect()
    )

    summary_lines = []
    summary_lines.append(f"Total log lines processed: {total_lines:,}")
    summary_lines.append(f"Total lines with log levels: {total_with_levels:,}")
    summary_lines.append(f"Unique log levels found: {unique_levels}")
    summary_lines.append("")
    summary_lines.append("Log level distribution:")

    for row in level_distribution:
        level = row['level']
        count = row['count']
        percentage = (count / total_with_levels) * 100
        summary_lines.append(f"  {level:<6}: {count:>12,} ({percentage:>6.2f}%)")
    
    # Write to file
    output_file = 'data/output/problem1_summary_local.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))


def main():
    if len(sys.argv) > 1:
        master_url = sys.argv[1]
    else:
        # Try to get from environment variable
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("❌ Error: Master URL not provided")
            print("Usage: python nyc_tlc_problem1_cluster.py spark://MASTER_IP:7077")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            return 1
    spark = create_spark_session(master_url)

    # Load all log files
    try:
        s3_path = "s3a://spv15-assignment-spark-cluster-logs/data/*/*.log"
        logs_df = spark.read.text(s3_path)
        total_lines = logs_df.count()
        success = True

    except Exception as e:
        print(f"Error loading log files: {str(e)}")
        success = False

    # Parse log files
    try:
        parsed_logs = parse_logs(logs_df)
        total_with_levels = parsed_logs.filter(col('level') != '').count()
        unique_levels = parsed_logs.filter(col('level') != '').select('level').distinct().count()
    except Exception as e:
        print(f"Error parsing log files: {str(e)}")
        success = False

    # Get level counts
    try:
        level_counts(parsed_logs)
    except Exception as e:
        print(f"Error getting level counts: {str(e)}")
        success = False

    # Get random sample
    try:
        get_random(parsed_logs)
    except Exception as e:
        print(f"Error getting random rows: {str(e)}")
        success = False

    # Get file and analysis summary
    try:
        get_summary(total_lines, total_with_levels, unique_levels, parsed_logs)
    except Exception as e:
        print(f"Error getting summary: {str(e)}")
        success = False

    # Clean up
    spark.stop()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())