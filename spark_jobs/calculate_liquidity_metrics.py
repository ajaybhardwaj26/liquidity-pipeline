import logging
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, to_timestamp
from pyspark.sql.window import Window

# Set up logging
logging.basicConfig(level=logging.INFO)

def read_raw_data(spark, s3_path):
    """Reads the raw CSV data from S3"""
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(s3_path)

def clean_and_transform(df):
    """Cleans and transforms the raw data"""
    df = df.withColumn("timestamp", to_timestamp("timestamp", "M/d/yyyy H:mm"))

    window_spec = Window.partitionBy("symbol").orderBy("timestamp")

    df = df.withColumn("prev_price", lag("price").over(window_spec))
    df = df.withColumn("price_change", col("price") - col("prev_price"))
    df = df.withColumn("liquidity_cost", col("price_change") * col("volume"))

    return df

def write_to_s3(df, output_path):
    """Writes the processed data to S3"""
    df.write.mode("overwrite").parquet(output_path)

def wait_for_job_completion(spark):
    """Wait for all jobs to complete before stopping Spark"""
    while True:
        active_jobs = spark.sparkContext.statusTracker().getJobIdsForGroup(None)
        if len(active_jobs) == 0:  # No active jobs
            break
        logging.info(f"Active jobs: {active_jobs}. Waiting for completion...")
        time.sleep(1)  # Poll every second
    logging.info("All Spark jobs completed.")

def main():
    """Main function to execute the data pipeline"""
    s3_input = "s3a://liquidity-pipeline-data/raw/market_feed/market_feed_sample.csv"
    s3_output = "s3a://liquidity-pipeline-data/processed/liquidity_metrics/"

    try:
        logging.info("Starting Spark session...")
        # Start Spark session
        spark = SparkSession.builder.appName("LiquidityPipeline").getOrCreate()

        # Read the raw data
        logging.info("Reading raw data from S3...")
        raw_df = read_raw_data(spark, s3_input)

        # Display raw timestamps (for debugging)
        logging.info("Printing raw timestamps...")
        rows = raw_df.select("timestamp").collect()
        for row in rows:
            logging.info(f"Raw timestamp value: {row['timestamp']}")

        # Process the data
        logging.info("Cleaning and transforming data...")
        processed_df = clean_and_transform(raw_df)

        # Write the processed data to S3
        logging.info("Writing processed data to S3...")
        write_to_s3(processed_df, s3_output)
        logging.info("Data written to S3 successfully.")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())

    finally:
        logging.info("Clearing cache...")
        spark.catalog.clearCache()  # Clear lingering data in memory

        logging.info("Waiting for all jobs to finish before stopping Spark...")
        wait_for_job_completion(spark)  # Wait for jobs to complete

        logging.info("Stopping Spark session...")
        spark.stop()  # Stop the Spark session gracefully
        logging.info("Spark session stopped.")

        # Do not call System.exit() here, as it causes issues with Py4J in Spark.
        # Just let Python exit naturally.

if __name__ == "__main__":
    main()
