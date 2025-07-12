import time
import logging
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

def main():
    """Main function to execute the data pipeline"""
    s3_input = "s3a://liquidity-pipeline-data/raw/market_feed/market_feed_sample.csv"
    s3_output = "s3a://liquidity-pipeline-data/processed/liquidity_metrics/"

    try:
        # Initialize the Spark session
        logging.info("Starting Spark session...")
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
        # Optionally log the full stack trace
        import traceback
        logging.error(traceback.format_exc())

    finally:
        # Clear cache and stop the Spark session
        logging.info("Clearing cache...")
        spark.catalog.clearCache()  # Clear lingering data in memory

        logging.info("Stopping Spark session...")
        time.sleep(2)  # Wait for logs to appear before stopping Spark

        # Gracefully stop the Spark session
        logging.info("Stopping Spark...")
        spark.stop()
        logging.info("Spark stopped.")

if __name__ == "__main__":
    main()