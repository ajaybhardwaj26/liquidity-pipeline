from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, to_timestamp, expr
from pyspark.sql.window import Window
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
    s3_input = "s3a://liquidity-pipeline-data/raw/market_feed/market_feed_sample.csv"
    s3_output = "s3a://liquidity-pipeline-data/processed/liquidity_metrics/"

    spark = SparkSession.builder.appName("LiquidityPipeline").getOrCreate()

    raw_df = read_raw_data(spark, s3_input)

    rows = raw_df.select("timestamp").collect()
    for row in rows:
        print(f"Raw timestamp value: {row['timestamp']}")


    processed_df = clean_and_transform(raw_df)
    write_to_s3(processed_df, s3_output)
    print(f"Data written to {output_path}.")

    spark.stop()
    print(f"spark stopped")
if __name__ == "__main__":
    main()