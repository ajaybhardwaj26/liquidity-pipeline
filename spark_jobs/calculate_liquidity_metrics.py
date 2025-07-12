import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, to_timestamp, expr
from pyspark.sql.window import Window
import logging
def read_raw_data(spark, s3_path):
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(s3_path)

def clean_and_transform(df):
    df = df.withColumn("timestamp", to_timestamp("timestamp", "M/d/yyyy H:mm"))

    window_spec = Window.partitionBy("symbol").orderBy("timestamp")

    df = df.withColumn("prev_price", lag("price").over(window_spec))
    df = df.withColumn("price_change", col("price") - col("prev_price"))
    df = df.withColumn("liquidity_cost", col("price_change") * col("volume"))

    return df

def write_to_s3(df, output_path):
    df.write.mode("overwrite").parquet(output_path)

def main():
    logging.basicConfig(level=logging.INFO)
    s3_input = "s3a://liquidity-pipeline-data/raw/market_feed/market_feed_sample.csv"
    s3_output = "s3a://liquidity-pipeline-data/processed/liquidity_metrics/"

    # Start Spark session
    logging.info("Starting Spark session...")
    spark = SparkSession.builder.appName("LiquidityPipeline").getOrCreate()

    try:
        raw_df = read_raw_data(spark, s3_input)

        rows = raw_df.select("timestamp").collect()
        for row in rows:
            print(f"Raw timestamp value: {row['timestamp']}")

        processed_df = clean_and_transform(raw_df)
        write_to_s3(processed_df, s3_output)
        logging.info("Data written to S3")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

    finally:
        logging.info("Clearing cache...")
        spark.catalog.clearCache()  # Clear any lingering data in memory

        logging.info("Stopping Spark session...")
        time.sleep(2)  # Add delay to allow print messages to appear before stopping Spark
        spark.stop()
        sc = spark.sparkContext
        sc.stop()  # Force stop SparkContext
        logging.info("Spark session stopped")

if __name__ == "__main__":
    main()